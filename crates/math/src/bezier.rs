use crate::prelude::*;

pub trait Bezier: Clone + Copy {
    /// Get the point on the curve at `t`.
    fn point(self, t: f32) -> Vec2;

    /// Get the tangent vector of the curve at a given `t`. Not guaranteed to be
    /// normalized.
    fn tangent(self, t: f32) -> Vec2;

    /// Get the bounding box of the curve. May be conservative.
    fn bounding_box(self) -> Rect;

    /// Scale the curve.
    fn scale(self, scale: f32) -> Self ;

    /// Translate the curve.
    fn translate(self, delta: Vec2) -> Self;
}

/// A linear bezier curve.
#[derive(Debug, Clone, Copy)]
pub struct Line {
    /// The start point of the curve.
    pub p0: Vec2,

    /// The end point of the curve.
    pub p1: Vec2,
}

impl Line {
    /// Create a line going from `p0` to `p1`.
    pub fn new(p0: Vec2, p1: Vec2) -> Self {
        Self { p0, p1 }
    }
}

impl Bezier for Line {
    /// Get the point at `t` on the line.
    #[inline]
    #[must_use]
    fn point(self, t: f32) -> Vec2 {
        self.p0.lerp(self.p1, t) 
    }

    /// Get the tangent of the line.
    #[inline]
    #[must_use]
    fn tangent(self, _: f32) -> Vec2 {
        self.p1 - self.p0
    }

    /// Get the bounding box of the line.
    #[inline]
    #[must_use]
    fn bounding_box(self) -> Rect {
        let min = Vec2::min(self.p0, self.p1);
        let max = Vec2::max(self.p0, self.p1);

        Rect::from_corners(min, max)
    }

    /// Returns a line scaled to `scale`.
    #[inline]
    #[must_use]
    fn scale(mut self, scale: f32) -> Self {
        self.p0 *= scale;
        self.p1 *= scale;

        self
    }

    /// Returns a line translated by `delta`.
    #[inline]
    #[must_use]
    fn translate(mut self, delta: Vec2) -> Self {
        self.p0 += delta;
        self.p1 += delta;

        self
    }
}

/// A quadratic bezier curve.
#[derive(Debug, Clone, Copy)]
pub struct Quadratic {
    /// The start point.
    pub p0: Vec2,

    /// The control point.    
    pub p1: Vec2,

    /// The end point.
    pub p2: Vec2,
}

impl Quadratic {
    /// Create a quadratic bezier curve with the points `p0`, `p1` and `p2`.
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2) -> Self {
        Self { p0, p1, p2 }
    }
}

impl Bezier for Quadratic {
    /// The the point at `t` along the curve.
    #[inline]
    #[must_use]
    fn point(self, t: f32) -> Vec2 {
        let a = self.p0.lerp(self.p1, t);
        let b = self.p1.lerp(self.p2, t);

        a.lerp(b, t)
    }

    /// Get the tangent of the curve at `t`.
    #[inline]
    #[must_use]
    fn tangent(self, t: f32) -> Vec2 {
        let p0_p1 = self.p1 - self.p0;
        let p1_p2 = self.p2 - self.p1;
       
        p0_p1.lerp(p1_p2, t)
    }

    /// Get the bounding box the of curve.
    #[inline]
    #[must_use]
    fn bounding_box(self) -> Rect {
        let rect = Rect::from_corners(
            Vec2::min(self.p0, self.p2),
            Vec2::max(self.p0, self.p2),
        );

        if !rect.encloses_point(self.p1) {
            let t = (self.p0 - self.p1) / (self.p0 - 2.0 * self.p1 + self.p2);
            let t = t.clamp(Vec2::ZERO, Vec2::splat(1.0));

            let s = 1.0 - t;
            let q = s * s * self.p0 + 2.0 * s * t * self.p1 + t * t * self.p2;

            Rect::from_corners(
                Vec2::min(rect.min, q),
                Vec2::max(rect.max, q),
            )
        } else {
            rect
        }
    }

    /// Returns the curve scaled by `scale`.
    #[inline]
    #[must_use]
    fn scale(mut self, scale: f32) -> Self {
        self.p0 *= scale;
        self.p1 *= scale;
        self.p2 *= scale;

        self
    }

    /// Returns the curve translated by `delta`.
    #[inline]
    #[must_use]
    fn translate(mut self, delta: Vec2) -> Self {
        self.p0 += delta;
        self.p1 += delta;
        self.p2 += delta;

        self
    }
}

/// A cubic bezier curve.
#[derive(Debug, Clone, Copy)]
pub struct Cubic {
    /// The start point.
    pub p0: Vec2,

    /// The first control point.
    pub p1: Vec2,

    /// The second control point.
    pub p2: Vec2,

    /// The end point.
    pub p3: Vec2,
}

impl Cubic {
    /// Create a cubic bezier curve with the points `p0`, `p1`, `p2` and `p3`.
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        Self { p0, p1, p2, p3 }
    }
}

impl Bezier for Cubic {
    /// Get the point at `t` along the curve.
    #[inline]
    #[must_use]
    fn point(self, t: f32) -> Vec2 {
        let a = self.p0.lerp(self.p1, t);
        let b = self.p1.lerp(self.p2, t); 
        let c = self.p2.lerp(self.p3, t);

        let a = a.lerp(b, t);
        let b = b.lerp(c, t);

        a.lerp(b, t)
    }

    /// Get the tangent of the curve at `t`.
    #[inline]
    #[must_use]
    fn tangent(self, t: f32) -> Vec2 {
        let p0_p1 = self.p1 - self.p0; 
        let p1_p2 = self.p2 - self.p1; 
        let p2_p3 = self.p3 - self.p2;

        let a = p0_p1.lerp(p1_p2, t);
        let b = p1_p2.lerp(p2_p3, t);

        a.lerp(b, t)
    }

    /// Get the bounding box of the curve.
    #[inline]
    #[must_use]
    fn bounding_box(self) -> Rect {
        // TODO: Make this non-conservative like the others.

        let min = Vec2::min(self.p0, Vec2::min(self.p1, Vec2::min(self.p2, self.p3)));
        let max = Vec2::max(self.p0, Vec2::max(self.p1, Vec2::max(self.p2, self.p3)));

        Rect::from_corners(min, max)
    }

    /// Returns the curve scaled by `scale`.
    #[inline]
    #[must_use]
    fn scale(mut self, scale: f32) -> Self {
        self.p0 *= scale;
        self.p1 *= scale;
        self.p2 *= scale;
        self.p3 *= scale;

        self
    }

    /// Returns the curve translated by `delta`.
    #[inline]
    #[must_use]
    fn translate(mut self, delta: Vec2) -> Self {
        self.p0 += delta;
        self.p1 += delta;
        self.p2 += delta;
        self.p3 += delta;

        self
    }
}

