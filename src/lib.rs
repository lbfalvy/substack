//! A linked list on the stack to avoid heap allocations in recursive algorithms
//!
//! ```
//! use substack::Substack;
//!
//! /// Walk a disjoint set and find the representative of an element, or return None if the
//! /// set contains a loop
//! fn find_value(alias_map: &[usize], idx: usize, prev: Substack<usize>) -> Option<usize> {
//!   match () {
//!     () if alias_map[idx] == idx => Some(idx),
//!     () if prev.iter().any(|i| *i == idx) => None,
//!     () => find_value(alias_map, alias_map[idx], prev.push(idx)),
//!   }
//! }
//!
//! const map: &[usize] = &[2, 4, 1, 5, 1, 5];
//!
//! assert_eq!(find_value(map, 0, Substack::Bottom), None);
//! assert_eq!(find_value(map, 3, Substack::Bottom), Some(5));
//! ```

use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Deref;

/// A frame of [Substack]
#[derive(Clone, Copy)]
pub struct Stackframe<'a, T> {
  /// The item held in this frame
  pub item: T,
  /// Lower stack frames
  pub prev: &'a Substack<'a, T>,
  /// The length of the stack up to and including this layer
  pub len: usize,
}

impl<'a, T> Deref for Stackframe<'a, T> {
  type Target = T;
  fn deref(&self) -> &Self::Target { &self.item }
}

/// A FILO stack that lives on the regular call stack as a linked list.
#[derive(Clone, Copy)]
pub enum Substack<'a, T> {
  /// A level in the linked list
  Frame(Stackframe<'a, T>),
  /// The end of the list
  Bottom,
}

impl<'a, T> Substack<'a, T> {
  /// Convert the substack into an option of stackframe
  pub fn opt(&'a self) -> Option<&'a Stackframe<'a, T>> {
    match self {
      Self::Frame(f) => Some(f),
      Self::Bottom => None,
    }
  }

  /// Construct an iterator starting from the top and moving down the stack
  pub fn iter(&self) -> SubstackIterator<T> { SubstackIterator { curr: self } }

  /// Add the item to this substack
  pub fn push(&'a self, item: T) -> Self { Self::Frame(self.new_frame(item)) }

  /// Create a new frame on top of this substack
  pub fn new_frame(&'a self, item: T) -> Stackframe<'a, T> {
    Stackframe { item, prev: self, len: self.opt().map_or(1, |s| s.len + 1) }
  }

  /// obtain the previous stackframe if one exists.
  ///
  /// # Panics
  ///
  /// if the index is greater than the number of stack frames
  pub fn pop(&'a self, count: usize) -> &'a Self {
    match (self, count) {
      (_, 0) => self,
      (Self::Frame(f), _) => f.prev.pop(count - 1),
      (Self::Bottom, _) => panic!("Stack index out of bounds"),
    }
  }

  /// number of stackframes
  pub fn len(&self) -> usize {
    match self {
      Self::Frame(f) => f.len,
      Self::Bottom => 0,
    }
  }

  /// is this the bottom of the stack
  pub fn is_empty(&self) -> bool { self.len() == 0 }

  /// Get a reference to the value held in this stackframe
  pub fn value(&self) -> Option<&T> { self.opt().map(|f| &f.item) }

  /// Clones the elements into a vector starting at the bottom of the stack
  pub fn unreverse(&self) -> Vec<T>
  where
    T: Clone,
  {
    self.iter().unreverse()
  }

  /// Visit all of the elements lowest first. This uses internal recursion, but
  /// since the iterator itself fits on the stack it's very likely that the
  /// reversed slices to it also will. If the callback is pure, these stack
  /// frames should store a single reference to the corresponding item.
  pub fn rfold<'b, U, F: FnMut(U, &'b T) -> U>(&'b self, default: U, mut callback: F) -> U {
    self.iter().rfold_rec(default, &mut callback)
  }
}

impl<'a, T: Debug> Debug for Substack<'a, T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Substack")?;
    f.debug_list().entries(self.iter()).finish()
  }
}

/// Iterates over a substack from the top down
pub struct SubstackIterator<'a, T> {
  curr: &'a Substack<'a, T>,
}

impl<'a, T> SubstackIterator<'a, T> {
  /// Clones the elements out to a vector which starts at the bottom of the
  /// stack
  pub fn unreverse(self) -> Vec<T>
  where
    T: Clone,
  {
    let mut deque = VecDeque::with_capacity(self.curr.len());
    for item in self {
      deque.push_front(item.clone())
    }
    deque.into()
  }

  fn rfold_rec<U>(mut self, default: U, callback: &mut impl FnMut(U, &'a T) -> U) -> U {
    match self.next() {
      None => default,
      Some(t) => {
        let rec = self.rfold_rec(default, callback);
        callback(rec, t)
      },
    }
  }

  /// Visit all of the elements lowest first. This uses internal recursion, but
  /// since the iterator itself fits on the stack it's very likely that the
  /// reversed slices to it also will. If the callback is pure, these stack
  /// frames should store a single reference to the corresponding item.
  pub fn rfold<U, F: FnMut(U, &'a T) -> U>(self, default: U, mut callback: F) -> U {
    self.rfold_rec(default, &mut callback)
  }
}

impl<'a, T> Copy for SubstackIterator<'a, T> {}
impl<'a, T> Clone for SubstackIterator<'a, T> {
  fn clone(&self) -> Self { *self }
}

impl<'a, T> Iterator for SubstackIterator<'a, T> {
  type Item = &'a T;
  fn next(&mut self) -> Option<&'a T> {
    let curr = self.curr.opt()?;
    let item = &curr.item;
    let prev = curr.prev;
    self.curr = prev;
    Some(item)
  }

  fn size_hint(&self) -> (usize, Option<usize>) { (self.curr.len(), Some(self.curr.len())) }
}

/// Recursively walk an iterator, building a Substack from its items.
///
/// # Aborts
///
/// If the iterator is long, this will overflow the stack. Be very careful with
/// calling it on iterators of unknown length, or in an already recursive
/// context.
pub fn with_iter_stack<T, I: IntoIterator<Item = T>, F: FnOnce(Substack<T>) -> U, U>(
  iter: I,
  callback: F,
) -> U {
  with_iter_stack_rec(iter.into_iter(), callback, Substack::Bottom)
}

/// # stack utilization
///
/// iter and callback are moved out, so the only items that remain on the stack
/// are the function frame and substack
fn with_iter_stack_rec<T, I: Iterator<Item = T>, F: FnOnce(Substack<T>) -> U, U>(
  mut iter: I,
  callback: F,
  substack: Substack<T>,
) -> U {
  match iter.next() {
    None => callback(substack),
    Some(t) => with_iter_stack_rec(iter, callback, substack.push(t)),
  }
}

#[cfg(test)]
mod test {
  use crate::{Substack, with_iter_stack};

  // fill a stack with numbers from n to 0, then call the callback with it
  fn descending_ints(num: usize, stack: Substack<usize>, then: impl FnOnce(Substack<usize>)) {
    match num {
      0 => then(stack.push(0)),
      n => descending_ints(n - 1, stack.push(n), then),
    }
  }

  #[test]
  fn general() {
    descending_ints(5, Substack::Bottom, |nums| {
      let rev_items = nums.iter().cloned().collect::<Vec<_>>();
      assert_eq!(rev_items, [0, 1, 2, 3, 4, 5], "iterator is reversed");
      let asc_string = nums.rfold(String::new(), |s, d| s + &d.to_string());
      assert_eq!(asc_string, "543210", "rfold visits in reverse order");
      assert_eq!(nums.len(), 6, "length is correct");
      assert!(!nums.is_empty(), "is not empty");
      assert_eq!(nums.unreverse(), [5, 4, 3, 2, 1, 0], "unreverse");
      assert_eq!(nums.pop(0).value(), Some(&0), "popping none");
      assert_eq!(nums.pop(2).value(), Some(&2), "popping multiple");
      assert!(matches!(nums.pop(6), Substack::Bottom), "popping all");
    })
  }

  #[test]
  #[should_panic]
  fn out_of_bounds() {
    descending_ints(5, Substack::Bottom, |nums| {
      nums.pop(7);
    })
  }

  #[test]
  fn empty() {
    let b = Substack::Bottom::<()>;
    assert_eq!(b.len(), 0, "length computes");
    assert!(b.is_empty(), "is empty");
    assert_eq!(b.pop(0).value(), None, "can pop nothing from empty without panic");
  }

  #[test]
  fn with_iter() {
    let output = with_iter_stack([1, 2, 3, 4], |substack| {
      assert_eq!(substack.len(), 4);
      assert_eq!(substack.iter().nth(2), Some(&2));
      3
    });
    assert_eq!(output, 3);
  }
}
