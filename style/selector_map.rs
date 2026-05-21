/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! A data structure to efficiently index structs containing selectors by local
//! name, ids and hash.

use crate::applicable_declarations::{ApplicableDeclarationList, ScopeProximity};
use crate::context::QuirksMode;
use crate::data::ElementData;
use crate::dom::TElement;
use crate::sharing::StyleSharingElement;
use crate::rule_tree::CascadeLevel;
use crate::selector_parser::SelectorImpl;
use crate::stylesheets::scope_rule::ImplicitScopeRoot;
use crate::stylist::{CascadeData, ContainerConditionId, Rule, ScopeConditionId, Stylist};
use crate::AllocErr;
use crate::values::AtomIdent;
use crate::values::computed::Display;
use crate::{Atom, LocalName, Namespace, ShrinkIfNeeded, WeakAtom};
use atomic_refcell::AtomicRef;
use dom::ElementState;
use precomputed_hash::PrecomputedHash;
use selectors::{Element, OpaqueElement};
use selectors::matching::{MatchingContext, SelectorStats, Statistics, matches_selector};
use selectors::parser::{Combinator, Component, Selector, SelectorIter};
use smallvec::SmallVec;
use std::collections::hash_map;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use tsc_timer::Start;

/// A hasher implementation that doesn't hash anything, because it expects its
/// input to be a suitable u32 hash.
pub struct PrecomputedHasher {
    hash: Option<u32>,
}

impl Default for PrecomputedHasher {
    fn default() -> Self {
        Self { hash: None }
    }
}

/// A vector of relevant attributes, that can be useful for revalidation.
pub type RelevantAttributes = thin_vec::ThinVec<LocalName>;

/// This is a set of pseudo-classes that are both relatively-rare (they don't
/// affect most elements by default) and likely or known to have global rules
/// (in e.g., the UA sheets).
///
/// We can avoid selector-matching those global rules for all elements without
/// these pseudo-class states.
const RARE_PSEUDO_CLASS_STATES: ElementState = ElementState::from_bits_retain(
    ElementState::FULLSCREEN.bits()
        | ElementState::VISITED_OR_UNVISITED.bits()
        | ElementState::URLTARGET.bits()
        | ElementState::INERT.bits()
        | ElementState::FOCUS.bits()
        | ElementState::FOCUSRING.bits()
        | ElementState::TOPMOST_MODAL.bits()
        | ElementState::SUPPRESS_FOR_PRINT_SELECTION.bits()
        | ElementState::HEADING_LEVEL_BITS.bits(),
);

/// A simple alias for a hashmap using PrecomputedHasher.
pub type PrecomputedHashMap<K, V> = HashMap<K, V, BuildHasherDefault<PrecomputedHasher>>;

/// A simple alias for a hashset using PrecomputedHasher.
pub type PrecomputedHashSet<K> = HashSet<K, BuildHasherDefault<PrecomputedHasher>>;

impl Hasher for PrecomputedHasher {
    #[inline]
    fn write(&mut self, _: &[u8]) {
        unreachable!(
            "Called into PrecomputedHasher with something that isn't \
             a u32"
        )
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        debug_assert!(self.hash.is_none());
        self.hash = Some(i);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash.expect("PrecomputedHasher wasn't fed?") as u64
    }
}

/// A trait to abstract over a given selector map entry.
pub trait SelectorMapEntry: Sized + Clone {
    /// Gets the selector we should use to index in the selector map.
    fn selector(&self) -> SelectorIter<'_, SelectorImpl>;
}

/// Map element data to selector-providing objects for which the last simple
/// selector starts with them.
///
/// e.g.,
/// "p > img" would go into the set of selectors corresponding to the
/// element "img"
/// "a .foo .bar.baz" would go into the set of selectors corresponding to
/// the class "bar"
///
/// Because we match selectors right-to-left (i.e., moving up the tree
/// from an element), we need to compare the last simple selector in the
/// selector with the element.
///
/// So, if an element has ID "id1" and classes "foo" and "bar", then all
/// the rules it matches will have their last simple selector starting
/// either with "#id1" or with ".foo" or with ".bar".
///
/// Hence, the union of the rules keyed on each of element's classes, ID,
/// element name, etc. will contain the Selectors that actually match that
/// element.
///
/// We use a 1-entry SmallVec to avoid a separate heap allocation in the case
/// where we only have one entry, which is quite common. See measurements in:
/// * https://bugzilla.mozilla.org/show_bug.cgi?id=1363789#c5
/// * https://bugzilla.mozilla.org/show_bug.cgi?id=681755
///

/// A trait providing the necessary functionality for an element type to work
/// with a SelectorMap.
pub trait SelectorMapElement: Element<Impl = SelectorImpl> + Copy + std::fmt::Debug {
    /// The ID for this element.
    fn id(&self) -> Option<&WeakAtom>;

    /// Internal iterator for the classes of this element.
    fn each_class<F>(&self, callback: F)
    where
        F: FnMut(&AtomIdent);

    /// Internal iterator for the attribute names of this element.
    fn each_attr_name<F>(&self, callback: F)
    where
        F: FnMut(&LocalName);

    /// Returns element's local name.
    fn local_name(&self) -> &<SelectorImpl as selectors::parser::SelectorImpl>::BorrowedLocalName;

    /// Get this element's state, for non-tree-structural pseudos.
    fn state(&self) -> ElementState;

    /// Returns element's namespace.
    fn namespace(&self)
        -> &<SelectorImpl as selectors::parser::SelectorImpl>::BorrowedNamespaceUrl;

    /// Returns the implicit scope root for given sheet index and host.
    fn implicit_scope_for_sheet_in_shadow_root(
        _opaque_host: OpaqueElement,
        _sheet_index: usize,
    ) -> Option<ImplicitScopeRoot> {
        None
    }

    /// Get this node's parent element from the perspective of a restyle
    /// traversal.
    fn traversal_parent(&self) -> Option<Self>;

    /// Immutably borrows the ElementData.
    fn borrow_data(&self) -> Option<AtomicRef<'_, ElementData>>;

    /// Returns the size of the element to be used in container size queries.
    /// This will usually be the size of the content area of the primary box,
    /// but can be None if there is no box or if some axis lacks size containment.
    fn query_container_size(
        &self,
        display: &Display,
    ) -> euclid::default::Size2D<Option<app_units::Au>>;
}
/// TODO: Tune the initial capacity of the HashMap
#[derive(Clone, Debug, MallocSizeOf)]
pub struct SelectorMap<T: 'static> {
    /// Rules that have `:root` selectors.
    pub root: SmallVec<[T; 1]>,
    /// A hash from an ID to rules which contain that ID selector.
    pub id_hash: MaybeCaseInsensitiveHashMap<Atom, SmallVec<[T; 1]>>,
    /// Rules with selectors that have common pseudo-classes such as :hover.
    pub common_pseudo_classes: SmallVec<[T; 1]>,
    /// A hash from a class name to rules which contain that class selector.
    pub class_hash: MaybeCaseInsensitiveHashMap<Atom, SmallVec<[T; 1]>>,
    /// A hash from local name to rules which contain that local name selector.
    pub local_name_hash: PrecomputedHashMap<LocalName, SmallVec<[T; 1]>>,
    /// A hash from attributes to rules which contain that attribute selector.
    pub attribute_hash: PrecomputedHashMap<LocalName, SmallVec<[T; 1]>>,
    /// A hash from namespace to rules which contain that namespace selector.
    pub namespace_hash: PrecomputedHashMap<Namespace, SmallVec<[T; 1]>>,
    /// Rules for pseudo-states that are rare but have global selectors.
    pub rare_pseudo_classes: SmallVec<[T; 1]>,
    /// All other rules.
    pub other: SmallVec<[T; 1]>,
    /// The number of entries in this map.
    pub count: usize,
}

impl<T: 'static> Default for SelectorMap<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SelectorMap<T> {
    /// Trivially constructs an empty `SelectorMap`.
    pub fn new() -> Self {
        SelectorMap {
            root: SmallVec::new(),
            id_hash: MaybeCaseInsensitiveHashMap::new(),
            common_pseudo_classes: SmallVec::new(),
            class_hash: MaybeCaseInsensitiveHashMap::new(),
            attribute_hash: HashMap::default(),
            local_name_hash: HashMap::default(),
            namespace_hash: HashMap::default(),
            rare_pseudo_classes: SmallVec::new(),
            other: SmallVec::new(),
            count: 0,
        }
    }

    /// Shrink the capacity of the map if needed.
    pub fn shrink_if_needed(&mut self) {
        self.id_hash.shrink_if_needed();
        self.class_hash.shrink_if_needed();
        self.attribute_hash.shrink_if_needed();
        self.local_name_hash.shrink_if_needed();
        self.namespace_hash.shrink_if_needed();
    }

    /// Clears the hashmap retaining storage.
    pub fn clear(&mut self) {
        self.root.clear();
        self.id_hash.clear();
        self.class_hash.clear();
        self.attribute_hash.clear();
        self.local_name_hash.clear();
        self.namespace_hash.clear();
        self.rare_pseudo_classes.clear();
        self.other.clear();
        self.count = 0;
    }

    /// Returns whether there are any entries in the map.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.count
    }
}

impl SelectorMap<Rule> {
    /// Append to `rule_list` all Rules in `self` that match element.
    ///
    /// Extract matching rules as per element's ID, classes, tag name, etc..
    /// Sort the Rules at the end to maintain cascading order.
    /// Return statistics
    pub fn get_all_matching_rules<E>(
        &self,
        element: E,
        rule_hash_target: E,
        matching_rules_list: &mut ApplicableDeclarationList,
        mut matching_selectors: Option<&mut SmallVec<[Selector<SelectorImpl> ; 16]>>,
        mut selector_stats: Option<&mut SmallVec<[(Selector<SelectorImpl>, SelectorStats); 16]>>,
        matching_context: &mut MatchingContext<E::Impl>,
        cascade_level: CascadeLevel,
        cascade_data: &CascadeData,
        stylist: &Stylist,
        debug_html_str: Option<&str>,
    ) -> Statistics
    where
        E: SelectorMapElement,
    {
        if self.is_empty() {
            return Statistics::default();
        }

        let quirks_mode = matching_context.quirks_mode();
        let mut hits = 0;
        let mut stats = Statistics::default();

        let start = Start::now();

        if rule_hash_target.is_root() {
            hits += self.root.len();
            stats += SelectorMap::get_matching_rules(
                element,
                &self.root,
                matching_rules_list,
                matching_selectors.as_deref_mut(),
                selector_stats.as_deref_mut(),
                matching_context,
                cascade_level,
                cascade_data,
                stylist,
                debug_html_str,
            );
        }

        if let Some(id) = rule_hash_target.id() {
            if let Some(rules) = self.id_hash.get(id, quirks_mode) {
                hits += rules.len();
                stats += SelectorMap::get_matching_rules(
                    element,
                    rules,
                    matching_rules_list,
                    matching_selectors.as_deref_mut(),
                    selector_stats.as_deref_mut(),
                    matching_context,
                    cascade_level,
                    cascade_data,
                    stylist,
                    debug_html_str,
                )
            }
        }

        // TODO: this is my best guess at how to determine if an element is eligible to be matched against all the selectors that have common pseudo-classes such as :hover.
        if !rule_hash_target.state().is_empty() {
            hits += self.common_pseudo_classes.len();
            stats += SelectorMap::get_matching_rules(
                element,
                &self.common_pseudo_classes,
                matching_rules_list,
                matching_selectors.as_deref_mut(),
                selector_stats.as_deref_mut(),
                matching_context,
                cascade_level,
                cascade_data,
                stylist,
                debug_html_str,
            );
        }

        rule_hash_target.each_class(|class| {
            if let Some(rules) = self.class_hash.get(&class, quirks_mode) {
                hits += rules.len();
                stats += SelectorMap::get_matching_rules(
                    element,
                    rules,
                    matching_rules_list,
                    matching_selectors.as_deref_mut(),
                    selector_stats.as_deref_mut(),
                    matching_context,
                    cascade_level,
                    cascade_data,
                    stylist,
                    debug_html_str,
                )
            }
        });

        rule_hash_target.each_attr_name(|name| {
            if let Some(rules) = self.attribute_hash.get(name) {
                hits += rules.len();
                stats += SelectorMap::get_matching_rules(
                    element,
                    rules,
                    matching_rules_list,
                    matching_selectors.as_deref_mut(),
                    selector_stats.as_deref_mut(),
                    matching_context,
                    cascade_level,
                    cascade_data,
                    stylist,
                    debug_html_str,
                )
            }
        });

        if let Some(rules) = self.local_name_hash.get(rule_hash_target.local_name()) {
            hits += rules.len();
            stats += SelectorMap::get_matching_rules(
                element,
                rules,
                matching_rules_list,
                matching_selectors.as_deref_mut(),
                selector_stats.as_deref_mut(),
                matching_context,
                cascade_level,
                cascade_data,
                stylist,
                debug_html_str,
            )
        }

        if rule_hash_target
            .state()
            .intersects(RARE_PSEUDO_CLASS_STATES)
        {
            hits += self.rare_pseudo_classes.len();
            stats += SelectorMap::get_matching_rules(
                element,
                &self.rare_pseudo_classes,
                matching_rules_list,
                matching_selectors.as_deref_mut(),
                selector_stats.as_deref_mut(),
                matching_context,
                cascade_level,
                cascade_data,
                stylist,
                debug_html_str,
            );
        }

        if let Some(rules) = self.namespace_hash.get(rule_hash_target.namespace()) {
            hits += rules.len();
            stats += SelectorMap::get_matching_rules(
                element,
                rules,
                matching_rules_list,
                matching_selectors.as_deref_mut(),
                selector_stats.as_deref_mut(),
                matching_context,
                cascade_level,
                cascade_data,
                stylist,
                debug_html_str,
            )
        }

        hits += self.other.len();
        stats += SelectorMap::get_matching_rules(
            element,
            &self.other,
            matching_rules_list,
            matching_selectors.as_deref_mut(),
            selector_stats.as_deref_mut(),
            matching_context,
            cascade_level,
            cascade_data,
            stylist,
            debug_html_str,
        );
        let duration = start.elapsed();
        stats.times.querying_selector_map = duration - stats.times._time_inside_buckets;
        stats.counts.selector_map_hits = hits;
        stats
    }

    /// Adds rules in `rules` that match `element` to the `matching_rules` list.
    /// Returns number of fast-rejects from the bloom filter.
    pub(crate) fn get_matching_rules<E>(
        element: E,
        rules: &[Rule],
        matching_rules: &mut ApplicableDeclarationList,
        mut matching_selectors: Option<&mut SmallVec<[Selector<SelectorImpl>; 16]>>,
        mut selector_stats: Option<&mut SmallVec<[(Selector<SelectorImpl>, SelectorStats); 16]>>,
        matching_context: &mut MatchingContext<E::Impl>,
        cascade_level: CascadeLevel,
        cascade_data: &CascadeData,
        stylist: &Stylist,
        debug_html_str: Option<&str>,
    ) -> Statistics
    where
        E: SelectorMapElement,
    {
        use selectors::matching::IncludeStartingStyle;

        let start = Start::now();
        let include_starting_style = matches!(
            matching_context.include_starting_style,
            IncludeStartingStyle::Yes
        );
        let mut acc_stats = Statistics::default();
        for rule in rules {
            #[cfg(feature = "debug_element")]
            if let Some(html_str) = debug_html_str {
                debug_element_selector(element, html_str, &rule.selector);
            }
            let scope_proximity = if rule.scope_condition_id == ScopeConditionId::none() {
                let (res, bloom_stats) = matches_selector(
                    &rule.selector,
                    0,
                    Some(&rule.hashes),
                    &element,
                    matching_context,
                );
                if let Some(selector_stats) = selector_stats.as_deref_mut() {
                    selector_stats.push((rule.selector.clone(), SelectorStats::Bloom(bloom_stats)));
                }
                acc_stats += bloom_stats;
                if !res {
                    continue;
                }
                ScopeProximity::infinity()
            } else {
                let (result, scope_proximity_stats) =
                    cascade_data.find_scope_proximity_if_matching(rule, element, matching_context);
                if let Some(selector_stats) = selector_stats.as_deref_mut() {
                    selector_stats.push((rule.selector.clone(), SelectorStats::ScopeProximity(scope_proximity_stats)));
                }
                acc_stats += scope_proximity_stats;
                if result == ScopeProximity::infinity() {
                    continue;
                }
                result
            };

            // TODO: figure out what stats to push for these if statements below the above one.
            if rule.container_condition_id != ContainerConditionId::none() {
                if !cascade_data.container_condition_matches(
                    rule.container_condition_id,
                    stylist,
                    element,
                    matching_context,
                ) {
                    continue;
                }
            }

            if rule.is_starting_style {
                // Set this flag if there are any rules inside @starting-style. This flag is for
                // optimization to avoid any redundant resolution of starting style if the author
                // doesn't specify for this element.
                matching_context.has_starting_style = true;

                if !include_starting_style {
                    continue;
                }
            }

            matching_rules.push(rule.to_applicable_declaration_block(
                cascade_level,
                cascade_data,
                scope_proximity,
            ));

            if let Some(matching_selectors) = matching_selectors.as_deref_mut() {
                matching_selectors.push(rule.selector.clone()); // TODO: Is cloning bad here?
            }
        }
        acc_stats.times._time_inside_buckets = start.elapsed();
        acc_stats
    }
}

impl<T: SelectorMapEntry> SelectorMap<T> {
    /// Inserts an entry into the correct bucket(s).
    pub fn insert(&mut self, entry: T, quirks_mode: QuirksMode) -> Result<(), AllocErr> {
        self.count += 1;

        // NOTE(emilio): It'd be nice for this to be a separate function, but
        // then the compiler can't reason about the lifetime dependency between
        // `entry` and `bucket`, and would force us to clone the rule in the
        // common path.
        macro_rules! insert_into_bucket {
            ($entry:ident, $bucket:expr) => {{
                if let Some(vec) = match $bucket {
                    Bucket::Root => Some(&mut self.root),
                    Bucket::ID(id) => Some(
                        self
                            .id_hash
                            .try_entry(id.clone(), quirks_mode)?
                            .or_default()
                    ),
                    Bucket::CommonPseudoClasses => Some(&mut self.common_pseudo_classes),
                    Bucket::Class(class) => Some(
                        self
                            .class_hash
                            .try_entry(class.clone(), quirks_mode)?
                            .or_default()
                    ),
                    Bucket::Attribute { name, lower_name }
                    | Bucket::LocalName { name, lower_name } => {
                        // If the local name in the selector isn't lowercase,
                        // insert it into the rule hash twice. This means that,
                        // during lookup, we can always find the rules based on
                        // the local name of the element, regardless of whether
                        // it's an html element in an html document (in which
                        // case we match against lower_name) or not (in which
                        // case we match against name).
                        //
                        // In the case of a non-html-element-in-html-document
                        // with a lowercase localname and a non-lowercase
                        // selector, the rulehash lookup may produce superfluous
                        // selectors, but the subsequent selector matching work
                        // will filter them out.
                        let is_attribute = matches!($bucket, Bucket::Attribute { .. });
                        let hash = if is_attribute {
                            &mut self.attribute_hash
                        } else {
                            &mut self.local_name_hash
                        };
                        if name != lower_name {
                            hash.try_reserve(1)?;
                            let vec = hash.entry(lower_name.clone()).or_default();
                            vec.try_reserve(1)?;
                            vec.push($entry.clone());
                        }
                        hash.try_reserve(1)?;
                        Some(hash.entry(name.clone()).or_default())
                    },
                    Bucket::Namespace(url) => {
                        self.namespace_hash.try_reserve(1)?;
                        Some(self.namespace_hash.entry(url.clone()).or_default())
                    },
                    Bucket::RarePseudoClasses => Some(&mut self.rare_pseudo_classes),
                    Bucket::Universal => Some(&mut self.other),
                    Bucket::None => None,
                } {
                    vec.try_reserve(1)?;
                    vec.push($entry);
                }
            }};
        }

        let bucket = {
            let mut disjoint_buckets = SmallVec::new();
            let bucket = find_bucket(entry.selector(), &mut disjoint_buckets);

            // See if inserting this selector in multiple entries in the
            // selector map would be worth it. Consider a case like:
            //
            //   .foo:where(div, #bar)
            //
            // There, `bucket` would be `Class(foo)`, and disjoint_buckets would
            // be `[LocalName { div }, ID(bar)]`.
            //
            // Here we choose to insert the selector in the `.foo` bucket in
            // such a case, as it's likely more worth it than inserting it in
            // both `div` and `#bar`.
            //
            // This is specially true if there's any universal selector in the
            // `disjoint_selectors` set, at which point we'd just be doing
            // wasted work.
            if !disjoint_buckets.is_empty()
                && disjoint_buckets
                    .iter()
                    .all(|b| b.more_specific_than(&bucket))
            {
                for bucket in &disjoint_buckets {
                    let entry = entry.clone();
                    insert_into_bucket!(entry, *bucket);
                }
                return Ok(());
            }
            bucket
        };

        insert_into_bucket!(entry, bucket);
        Ok(())
    }

    /// Looks up entries by id, class, local name, namespace, and other (in
    /// order).
    ///
    /// Each entry is passed to the callback, which returns true to continue
    /// iterating entries, or false to terminate the lookup.
    ///
    /// Returns false if the callback ever returns false.
    ///
    /// FIXME(bholley) This overlaps with SelectorMap<Rule>::get_all_matching_rules,
    /// but that function is extremely hot and I'd rather not rearrange it.
    pub fn lookup<'a, E, F>(
        &'a self,
        element: E,
        quirks_mode: QuirksMode,
        relevant_attributes: Option<&mut RelevantAttributes>,
        f: F,
    ) -> bool
    where
        E: StyleSharingElement,
        F: FnMut(&'a T) -> bool,
    {
        self.lookup_with_state(
            element,
            element.state(),
            quirks_mode,
            relevant_attributes,
            f,
        )
    }

    #[inline]
    fn lookup_with_state<'a, E, F>(
        &'a self,
        element: E,
        element_state: ElementState,
        quirks_mode: QuirksMode,
        mut relevant_attributes: Option<&mut RelevantAttributes>,
        mut f: F,
    ) -> bool
    where
        E: StyleSharingElement,
        F: FnMut(&'a T) -> bool,
    {
        if element.is_root() {
            for entry in self.root.iter() {
                if !f(&entry) {
                    return false;
                }
            }
        }

        if let Some(id) = element.id() {
            if let Some(v) = self.id_hash.get(id, quirks_mode) {
                for entry in v.iter() {
                    if !f(&entry) {
                        return false;
                    }
                }
            }
        }

        let mut done = false;
        element.each_class(|class| {
            if done {
                return;
            }
            if let Some(v) = self.class_hash.get(class, quirks_mode) {
                for entry in v.iter() {
                    if !f(&entry) {
                        done = true;
                        return;
                    }
                }
            }
        });

        if done {
            return false;
        }

        element.each_attr_name(|name| {
            if done {
                return;
            }
            if let Some(v) = self.attribute_hash.get(name) {
                if let Some(ref mut relevant_attributes) = relevant_attributes {
                    relevant_attributes.push(name.clone());
                }
                for entry in v.iter() {
                    if !f(&entry) {
                        done = true;
                        return;
                    }
                }
            }
        });

        if done {
            return false;
        }

        if let Some(v) = self.local_name_hash.get(element.local_name()) {
            for entry in v.iter() {
                if !f(&entry) {
                    return false;
                }
            }
        }

        if let Some(v) = self.namespace_hash.get(element.namespace()) {
            for entry in v.iter() {
                if !f(&entry) {
                    return false;
                }
            }
        }

        if element_state.intersects(RARE_PSEUDO_CLASS_STATES) {
            for entry in self.rare_pseudo_classes.iter() {
                if !f(&entry) {
                    return false;
                }
            }
        }

        for entry in self.other.iter() {
            if !f(&entry) {
                return false;
            }
        }

        true
    }

    /// Performs a normal lookup, and also looks up entries for the passed-in
    /// id and classes.
    ///
    /// Each entry is passed to the callback, which returns true to continue
    /// iterating entries, or false to terminate the lookup.
    ///
    /// Returns false if the callback ever returns false.
    #[inline]
    pub fn lookup_with_additional<'a, E, F>(
        &'a self,
        element: E,
        quirks_mode: QuirksMode,
        additional_id: Option<&WeakAtom>,
        additional_classes: &[Atom],
        additional_states: ElementState,
        mut f: F,
    ) -> bool
    where
        E: TElement,
        F: FnMut(&'a T) -> bool,
    {
        // Do the normal lookup.
        if !self.lookup_with_state(
            element,
            element.state() | additional_states,
            quirks_mode,
            /* relevant_attributes = */ None,
            |entry| f(entry),
        ) {
            return false;
        }

        // Check the additional id.
        if let Some(id) = additional_id {
            if let Some(v) = self.id_hash.get(id, quirks_mode) {
                for entry in v.iter() {
                    if !f(&entry) {
                        return false;
                    }
                }
            }
        }

        // Check the additional classes.
        for class in additional_classes {
            if let Some(v) = self.class_hash.get(class, quirks_mode) {
                for entry in v.iter() {
                    if !f(&entry) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

enum Bucket<'a> {
    Universal,
    Namespace(&'a Namespace),
    RarePseudoClasses,
    LocalName {
        name: &'a LocalName,
        lower_name: &'a LocalName,
    },
    Attribute {
        name: &'a LocalName,
        lower_name: &'a LocalName,
    },
    Class(&'a Atom),
    CommonPseudoClasses,
    ID(&'a Atom),
    Root,
    None,
}

impl<'a> Bucket<'a> {
    /// root > id > class > local name > namespace > pseudo-classes > universal.
    #[inline]
    fn specificity(&self) -> usize {
        match *self {
            Bucket::Universal => 0,
            Bucket::Namespace(..) => 1,
            Bucket::RarePseudoClasses => 2,
            Bucket::LocalName { .. } => 3,
            Bucket::Attribute { .. } => 4,
            Bucket::Class(..) => 5,
            Bucket::CommonPseudoClasses => 6, // TODO: :hover should be more specific than class and less specific than ID, since it can match up to one element and its ancestor chain. What about the other non-rare pseudo-classes?
            Bucket::ID(..) => 7,
            Bucket::Root => 8,
            Bucket::None => 9,
        }
    }

    #[inline]
    fn more_or_equally_specific_than(&self, other: &Self) -> bool {
        self.specificity() >= other.specificity()
    }

    #[inline]
    fn more_specific_than(&self, other: &Self) -> bool {
        self.specificity() > other.specificity()
    }
}

type DisjointBuckets<'a> = SmallVec<[Bucket<'a>; 5]>;

fn specific_bucket_for<'a>(
    component: &'a Component<SelectorImpl>,
    disjoint_buckets: &mut DisjointBuckets<'a>,
) -> Bucket<'a> {
    match *component {
        Component::Root => Bucket::Root,
        Component::ID(ref id) => Bucket::ID(id),
        Component::Class(ref class) => Bucket::Class(class),
        Component::AttributeInNoNamespace { ref local_name, .. } => Bucket::Attribute {
            name: local_name,
            lower_name: local_name,
        },
        Component::AttributeInNoNamespaceExists {
            ref local_name,
            ref local_name_lower,
        } => Bucket::Attribute {
            name: local_name,
            lower_name: local_name_lower,
        },
        Component::AttributeOther(ref selector) => Bucket::Attribute {
            name: &selector.local_name,
            lower_name: &selector.local_name_lower,
        },
        Component::LocalName(ref selector) => Bucket::LocalName {
            name: &selector.name,
            lower_name: &selector.lower_name,
        },
        Component::Namespace(_, ref url) | Component::DefaultNamespace(ref url) => {
            Bucket::Namespace(url)
        },
        // ::slotted(..) isn't a normal pseudo-element, so we can insert it on
        // the rule hash normally without much problem. For example, in a
        // selector like:
        //
        //   div::slotted(span)::before
        //
        // It looks like:
        //
        //  [
        //    LocalName(div),
        //    Combinator(SlotAssignment),
        //    Slotted(span),
        //    Combinator::PseudoElement,
        //    PseudoElement(::before),
        //  ]
        //
        // So inserting `span` in the rule hash makes sense since we want to
        // match the slotted <span>.
        Component::Slotted(ref selector) => find_bucket(selector.iter(), disjoint_buckets),
        Component::Host(Some(ref selector)) => find_bucket(selector.iter(), disjoint_buckets),
        Component::Is(ref list) | Component::Where(ref list) => {
            if list.len() == 1 {
                find_bucket(list.slice()[0].iter(), disjoint_buckets)
            } else {
                for selector in list.slice() {
                    let bucket = find_bucket(selector.iter(), disjoint_buckets);
                    disjoint_buckets.push(bucket);
                }
                Bucket::Universal
            }
        },
        Component::NonTSPseudoClass(ref pseudo_class) =>
        {
            if pseudo_class
                .state_flag()
                .intersects(RARE_PSEUDO_CLASS_STATES)
            {
                Bucket::RarePseudoClasses
            } else {
                // The goal of this is to get `:hover` selectors into a highly specific bucket. We are seeing that these are slow, and we think this is happening because if an element is not actually `:hover`ed, it won't be known until the slow match phase. In other words, `:hover` is quite a specific selector, so we should put `:hover` selectors in a bucket so that they are only tested against hovered elements.
                // TODO: should this really be one bucket for all selectors with any of the common pseudo-classes?
                Bucket::CommonPseudoClasses
            }
        },
        Component::Invalid(_) => Bucket::None,
        _ => Bucket::Universal,
    }
}

/// Searches a compound selector from left to right, and returns the appropriate
/// bucket for it.
///
/// It also populates disjoint_buckets with dependencies from nested selectors
/// with any semantics like :is() and :where().
#[inline(always)]
fn find_bucket<'a>(
    mut iter: SelectorIter<'a, SelectorImpl>,
    disjoint_buckets: &mut DisjointBuckets<'a>,
) -> Bucket<'a> {
    let mut current_bucket = Bucket::Universal;

    loop {
        for ss in &mut iter {
            let new_bucket = specific_bucket_for(ss, disjoint_buckets);
            // NOTE: When presented with the choice of multiple specific selectors, use the
            // rightmost, on the assumption that that's less common, see bug 1829540.
            if new_bucket.more_or_equally_specific_than(&current_bucket) {
                current_bucket = new_bucket;
            }
        }

        // Effectively, pseudo-elements are ignored, given only state
        // pseudo-classes may appear before them.
        if iter.next_sequence() != Some(Combinator::PseudoElement) {
            break;
        }
    }

    current_bucket
}

/// Wrapper for PrecomputedHashMap that does ASCII-case-insensitive lookup in quirks mode.
#[derive(Clone, Debug, MallocSizeOf)]
pub struct MaybeCaseInsensitiveHashMap<K: PrecomputedHash + Hash + Eq, V>(PrecomputedHashMap<K, V>);

impl<V> Default for MaybeCaseInsensitiveHashMap<Atom, V> {
    #[inline]
    fn default() -> Self {
        MaybeCaseInsensitiveHashMap(PrecomputedHashMap::default())
    }
}

impl<V> MaybeCaseInsensitiveHashMap<Atom, V> {
    /// Empty map
    pub fn new() -> Self {
        Self::default()
    }

    /// Shrink the capacity of the map if needed.
    pub fn shrink_if_needed(&mut self) {
        self.0.shrink_if_needed()
    }

    /// HashMap::try_entry
    pub fn try_entry(
        &mut self,
        mut key: Atom,
        quirks_mode: QuirksMode,
    ) -> Result<hash_map::Entry<'_, Atom, V>, AllocErr> {
        if quirks_mode == QuirksMode::Quirks {
            key = key.to_ascii_lowercase()
        }
        self.0.try_reserve(1)?;
        Ok(self.0.entry(key))
    }

    /// HashMap::is_empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// HashMap::iter
    pub fn iter(&self) -> hash_map::Iter<'_, Atom, V> {
        self.0.iter()
    }

    /// HashMap::clear
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// HashMap::get
    pub fn get(&self, key: &WeakAtom, quirks_mode: QuirksMode) -> Option<&V> {
        if quirks_mode == QuirksMode::Quirks {
            self.0.get(&key.to_ascii_lowercase())
        } else {
            self.0.get(key)
        }
    }
}

#[cfg(feature = "debug_element")]
/// Debug an element and selector
/// INSTRUCTIONS: Give an element the .DEBUG_ME class,
/// and put :where(*, .DEBUG_ME) somewhere in a selector you want to debug
pub fn debug_element_selector<E>(element: E, html_str: &str, current_selector: &Selector<SelectorImpl>)
where
    E: Element<Impl = SelectorImpl>,
{
    use cssparser::ToCss as _;

    if element.has_class(&AtomIdent::from("DEBUG_ME"), selectors::attr::CaseSensitivity::CaseSensitive)
        && current_selector.to_css_string().contains("DEBUG_ME") {
        std::hint::black_box(()); // put breakpoint here
    }
}
