use crate::prelude::*;

/// Axis aligned 2D rectangle.
#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Rect {
    /// Min point of the rectangle.
    pub min: Vec2,

    /// Max point of the rectangle.
    pub max: Vec2,
}

impl Rect {
    /// [`Rect`] where both points are [`Vec2::ZERO`].
    pub const ZERO: Self = Self::from_corners(Vec2::ZERO, Vec2::ZERO);

    /// Create new [`Rect`] from corner points.
    #[inline]
    pub const fn from_corners(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Create new [`Rect`] from corner coordinates.
    #[inline]
    pub const fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            min: Vec2::new(x0, y0),
            max: Vec2::new(x1, y1),
        }
    }

    /// The width of the rectangle.
    ///
    /// Note that the width may be negative if `min` and `max` are in reverse order, that is if the
    /// `min.x` is greater than `max.x`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    /// assert_eq!(rect.width(), 1.0);
    /// ```
    #[inline]
    pub fn width(self) -> f32 {
        self.max.x - self.min.x
    }

    /// The height of the rectangle.
    ///
    /// Note that the height may be negative if `min` and `max` are in reverse order, that is if the
    /// `min.y` is greater than `max.y`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    /// assert_eq!(rect.height(), 1.0);
    /// ```
    #[inline]
    pub fn height(self) -> f32 {
        self.max.y - self.min.y
    }

    /// The area of the rectangle.
    ///
    /// Note that the area can be negative if `min` and `max` are swapped.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    /// assert_eq!(rect.area(), 1.0);
    /// ```
    #[inline]
    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    /// Translates the rectangle by `delta`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0).translate(0.5);
    ///
    /// assert_eq!(rect.min.x, 0.5);
    /// assert_eq!(rect.min.y, 0.5);
    ///
    /// assert_eq!(rect.max.x, 1.5);
    /// assert_eq!(rect.max.y, 1.5);
    /// ```
    #[inline]
    pub fn translate(mut self, delta: f32) -> Self {
        self.min += delta;
        self.max += delta;

        self
    }

    /// Splits rectangle vertically at `min_offset`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    /// let (left, right) = rect.split_vertically(0.6);
    ///
    /// assert!((left.width() - 0.6).abs() < 0.01);
    /// assert!((right.width() - 0.4).abs() < 0.01);
    /// ```
    #[inline]
    pub fn split_vertically(mut self, min_offset: f32) -> (Self, Self) {
        let x = self.min.x + min_offset;
        let right = Self::from_corners(Vec2::new(x, self.min.y), self.max);

        self.max.x = x;

        (self, right)
    }

    /// Splits the rectangle horizontally at `min_offset`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    /// let (top, bottom) = rect.split_horizontally(0.5);
    ///
    /// assert_eq!(top.height(), 0.5);
    /// assert_eq!(bottom.height(), 0.5);
    /// ```
    #[inline]
    pub fn split_horizontally(mut self, min_offset: f32) -> (Self, Self) {
        let y = self.min.y + min_offset;
        let bottom = Self::from_corners(Vec2::new(self.min.x, y), self.max);

        self.max.y = y;

        (self, bottom)
    }

    /// Get the center point the rectangle.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    /// use rendi_math::prelude::*;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    ///
    /// assert_eq!(rect.center_point(), Vec2::splat(0.5));
    /// ```
    #[inline]
    pub fn center_point(self) -> Vec2 {
        let (width, height) = (self.width(), self.height());

        Vec2::new(self.min.x + width * 0.5, self.min.y + height * 0.5)
    }

    /// Checks if `point` lies inside the rectangle or on the border.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    /// use rendi_math::prelude::*;
    ///
    /// let rect = Rect::new(0.0, 0.0, 1.0, 1.0);
    ///
    /// assert!(rect.encloses_point(Vec2::splat(0.5)));
    /// assert!(!rect.encloses_point(Vec2::splat(1.5)));
    /// ```
    #[inline]
    pub fn encloses_point(self, point: Vec2) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }

    /// Checks if `rect` lies completely inside the rectangle or on the border.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    /// use rendi_math::prelude::*;
    //
    /// let outer = Rect::new(0.0, 0.0, 10.0, 10.0);
    /// let inner = Rect::new(2.5, 2.5, 7.5, 7.5);
    ///
    /// assert!(outer.encloses_rect(inner));
    /// assert!(!inner.encloses_rect(outer));
    /// ```
    #[inline]
    pub fn encloses_rect(self, rect: Rect) -> bool {
        rect.min.cmpge(self.min).all() && rect.max.cmple(self.max).all()
    }

    /// Returns a rectangle exactly bounding both rectangles.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_math::rect::Rect;
    /// use rendi_math::prelude::*;
    //
    /// let a = Rect::new(0.0, 0.0, 10.0, 10.0);
    /// let b = Rect::new(2.5, 2.5, 11.0, 11.0);
    ///
    /// assert_eq!(a.bounding_rect(b), Rect::new(0.0, 0.0, 11.0, 11.0));
    /// ```
    #[inline]
    pub fn bounding_rect(self, other: Rect) -> Self {
        let min = Vec2::min(self.min, other.min);
        let max = Vec2::max(self.max, other.max);

        Self::from_corners(min, max)
    }
}

