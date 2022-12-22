use std::borrow::Borrow;
use std::ops::{Index, IndexMut};
use std::{mem, slice, vec};

/// Map of sorted key-value pairs.
///
/// Optimal for usages with under 100 elements and more frequent lookup than insert or
/// removeals.
///
/// Very simular to `SortedMap` in rustc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortedMap<K, V> {
    elements: Vec<(K, V)>,
}

impl<K, V> Default for SortedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> SortedMap<K, V> {
    /// Returns a new empty sorted map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }
}

impl<K: Ord, V> SortedMap<K, V> {
    /// Create from a `Vec` sorted by key without duplicates.
    ///
    /// The map will not work correctly if `elements` isn't correcly sorted.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_data_structs::SortedMap;
    ///
    /// let map = SortedMap::from_sorted(vec![
    ///     (0, "zero"),
    ///     (1, "one"),
    ///     (2, "two"),
    /// ]);
    ///
    /// assert_eq!(map[&1], "one");
    /// ```
    ///
    #[must_use]
    pub fn from_sorted(elements: Vec<(K, V)>) -> Self {
        debug_assert!(elements.is_sorted_by(|a, b| a.0.partial_cmp(&b.0)));

        Self { elements }
    }

    /// Create from an unsorted `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_data_structs::SortedMap;
    ///
    /// let map = SortedMap::from_unsorted(vec![
    ///     (1, "one"),
    ///     (0, "zero"),
    ///     (2, "two"),
    /// ]);
    ///
    /// assert_eq!(map[&1], "one");
    /// ```
    ///
    #[must_use]
    pub fn from_unsorted(mut elements: Vec<(K, V)>) -> Self {
        elements.sort_unstable_by(|&(ref a, _), &(ref b, _)| a.cmp(b));
        elements.dedup_by(|&mut (ref a, _), &mut (ref b, _)| a.cmp(b).is_eq());

        Self { elements }
    }

    /// Insert an element into the map.
    ///
    /// Worst case O(n) time complexity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_data_structs::SortedMap;
    ///
    /// let mut map = SortedMap::new();
    ///     
    /// map.insert(1, "one");
    /// map.insert(0, "zero");
    /// map.insert(2, "two");
    ///
    /// assert_eq!(map[&1], "one");
    /// ```
    ///
    #[inline]
    pub fn insert(&mut self, key: K, mut value: V) -> Option<V> {
        match self.lookup_index_for(&key) {
            Ok(index) => {
                let slot = unsafe { self.elements.get_unchecked_mut(index) };

                mem::swap(&mut slot.1, &mut value);

                Some(value)
            }
            Err(index) => {
                self.elements.insert(index, (key, value));

                None
            }
        }
    }

    /// Insert an element into the map.
    ///
    /// Worst case O(n) time complexity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_data_structs::SortedMap;
    ///
    /// let mut map = SortedMap::from_sorted(vec![
    ///     (0, "zero"),
    ///     (1, "one"),
    ///     (2, "two"),
    /// ]);
    ///
    /// map.remove(&1);
    ///
    /// assert!(map.get(&1).is_none());
    /// ```
    ///
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.lookup_index_for(key) {
            Ok(index) => Some(self.elements.remove(index).1),
            Err(_) => None,
        }
    }

    /// Get a reference to the element with `key`.
    ///
    /// Returns `None` if the element doesn't exist in the map.
    #[inline]
    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let index = self.lookup_index_for(key).ok()?;

        unsafe { Some(&self.elements.get_unchecked(index).1) }
    }

    /// Get a mutable reference to the element with `key`.
    ///
    /// Returns `None` if the element doesn't exist in the map.
    #[inline]
    #[must_use]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let index = self.lookup_index_for(key).ok()?;

        unsafe { Some(&mut self.elements.get_unchecked_mut(index).1) }
    }

    #[inline(always)]
    fn lookup_index_for<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.elements
            .binary_search_by(|&(ref x, _)| x.borrow().cmp(key))
    }

    /// Returns `true` if the map contains an element with `key`.
    #[inline]
    #[must_use]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Get sorted iterator of the keys.
    #[inline]
    #[must_use]
    pub fn keys(&self) -> impl Iterator<Item = &K> + ExactSizeIterator + DoubleEndedIterator {
        self.elements.iter().map(|&(ref k, _)| k)
    }

    /// Get sorted iterator of the values.
    #[inline]
    #[must_use]
    pub fn values(&self) -> impl Iterator<Item = &V> + ExactSizeIterator + DoubleEndedIterator {
        self.elements.iter().map(|&(_, ref v)| v)
    }

    /// Returns an iterator over the map with keys grouped by a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_data_structs::SortedMap;
    ///
    /// let mut map = SortedMap::from_sorted(vec![
    ///     (1, "one"),
    ///     (2, "two"),
    ///     (2, "two"),
    ///     (5, "five"),
    /// ]);
    ///
    /// let mut groups = map.group_by(|a, b| a == b);
    ///
    /// assert_eq!(groups.next().unwrap().len(), 1);
    /// assert_eq!(groups.next().unwrap().len(), 2);
    /// assert_eq!(groups.next().unwrap().len(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn group_by<F>(&self, pred: F) -> GroupBy<'_, K, V, F>
    where
        F: FnMut(&K, &K) -> bool,
    {
        GroupBy {
            elements: &self.elements,
            pred,
        }
    }
}

pub struct GroupBy<'a, K, V, F> {
    elements: &'a [(K, V)],
    pred: F,
}

impl<'a, K, V, F> Iterator for GroupBy<'a, K, V, F>
where
    K: Ord,
    F: FnMut(&K, &K) -> bool,
{
    type Item = &'a [(K, V)];

    fn next(&mut self) -> Option<Self::Item> {
        if self.elements.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.elements.windows(2);

            while let Some([(k1, _), (k2, _)]) = iter.next() {
                if (self.pred)(k1, k2) {
                    len += 1
                } else {
                    break;
                }
            }

            let (head, tail) = self.elements.split_at(len);
            self.elements = tail;

            Some(head)
        }
    }
}

impl<K, V> SortedMap<K, V> {
    /// Clear the map.
    #[inline]
    pub fn clear(&mut self) {
        self.elements.clear()
    }

    /// Get the length of the map.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns true if the map is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Returns true if the map is empty.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> slice::Iter<(K, V)> {
        self.elements.iter()
    }

    /// Get the map as a sorted slice of key-value pairs.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[(K, V)] {
        &self.elements
    }

    /// Get the maximum key-value pair.
    ///
    /// Returns `None` if the map is empty.
    #[inline]
    #[must_use]
    pub fn last_key_value(&self) -> Option<&(K, V)> {
        self.elements.last()
    }

    /// Get the minimum key-value pair.
    ///
    /// Returns `None` if the map is empty.
    #[inline]
    #[must_use]
    pub fn first_key_value(&self) -> Option<&(K, V)> {
        self.elements.first()
    }

    /// Make the map into a sorted `Vec`.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<(K, V)> {
        self.elements
    }
}

impl<'a, K, Q, V> Index<&'a Q> for SortedMap<K, V>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<'a, K, Q, V> IndexMut<&'a Q> for SortedMap<K, V>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    fn index_mut(&mut self, key: &Q) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for SortedMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let elements: Vec<_> = iter.into_iter().collect();

        Self::from_unsorted(elements)
    }
}

impl<K: Ord, V> IntoIterator for SortedMap<K, V> {
    type Item = (K, V);
    type IntoIter = vec::IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a SortedMap<K, V> {
    type Item = &'a (K, V);
    type IntoIter = slice::Iter<'a, (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
