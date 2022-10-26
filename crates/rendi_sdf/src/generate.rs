use ttf_parser::{Face, OutlineBuilder};
use anyhow::{anyhow, Result};

use rendi_math::prelude::*;
use rendi_math::bezier::*;
use rendi_math::polynom;

use crate::GenerateInfo;

#[derive(Clone, Copy)]
struct SignedDist {
    /// The orthogonality indicates how orthogonal the line from the point to the shape is width
    /// the shape itself. `1.0` means the two lines are orthogonal.
    orthogonality: f32,

    /// The signed distance itself. This represents the euclidean distance to the edge of a shape.
    ///
    /// If the sign is negative, it's the distance from inside the shape to the closest edge.
    /// If the sign is positive, it's the distance from outside the shape to the closest edge.
    distance: f32,
}

#[derive(Debug, Clone, Copy)]
enum Segment {
    Line(Line),
    Quadratic(Quadratic),
}

impl Segment {
    #[inline]
    fn bounding_box(&self) -> Rect {
        match self {
            Segment::Line(line) => line.bounding_box(),
            Segment::Quadratic(curve) => curve.bounding_box(),
        }
    }
   
    /// Get the signed distance from `px` to the closest point of the segment.
    #[inline]
    fn signed_dist(&self, px: Vec2) -> SignedDist {
        match self {
            Segment::Line(line) => {
                // The method for finding the signed distance to a line segment is to first find
                // the point where a line to `px` is perpendicular with the line segment. This is
                // always the shortest distance. The t-value is found through a formula derived
                // from finding the root of the derivative of `line.point(t) - px`. This is simply
                // clamped to fit it onto the segment.

                let p0_p1 = line.p1 - line.p0;  
                let p0_px = px - line.p0;

                let perpendicular = p0_px.dot(p0_p1) / p0_p1.length_squared();
                let t = perpendicular.clamp(0.0, 1.0);

                let pt = line.point(t);
                let pt_px = px - pt;

                let orthogonality = if pt_px != Vec2::ZERO {
                    p0_p1.normalize().perp_dot(pt_px.normalize())
                } else {
                    0.0
                };

                SignedDist {
                    distance: pt_px.length() * orthogonality.signum(),
                    orthogonality: orthogonality.abs(),
                }
            }
            Segment::Quadratic(curve) => {
                // This method is simular to the one described above for lines. The closest point
                // is once again guaranteed to be perpendicular to the point. Solving this comes
                // down to finding the roots of the derivative of `curve.point(t) - p`, which means
                // solving a cubic equation. This also means that there is more than one candidate.
                // The one with the closest distance will be chosen.

                let v0 = px - curve.p0;
                let v1 = curve.p1 - curve.p0;
                let v2 = curve.p2 - 2.0 * curve.p1 + curve.p0;

                let a = v2.length_squared();
                let b = 3.0 * v1.dot(v2);
                let c = 2.0 * v1.length_squared() - v2.dot(v0);
                let d = -v1.dot(v0);

                let mut best_dist_squared = f32::MAX;
                let mut best_t = 0.0;

                for t in polynom::cubic_roots(a, b, c, d).as_ref() {
                    let t = t.clamp(0.0, 1.0); 
                    let pt = curve.point(t);

                    let dist_squared = (px - pt).length_squared();

                    if dist_squared < best_dist_squared {
                        best_dist_squared = dist_squared;
                        best_t = t;
                    }
                }

                let pt = curve.point(best_t);
                let tangent = curve.tangent(best_t);

                let pt_px = px - pt;

                let orthogonality = if pt_px != Vec2::ZERO && tangent != Vec2::ZERO {
                    tangent.normalize().perp_dot(pt_px.normalize())
                } else {
                    0.0
                };

                SignedDist {
                    distance: best_dist_squared.sqrt() * orthogonality.signum(),
                    orthogonality: orthogonality.abs(),
                }
            }
        }
    }

    /// Returns the segment scaled by `scale`.
    #[inline]
    fn scale(mut self, scale: f32) -> Self {
        match &mut self {
            Segment::Line(line) => {
                *line = line.scale(scale);
            }
            Segment::Quadratic(curve) => {
                *curve = curve.scale(scale);
            }
        }

        self
    }

    /// Returns the segment translated by `delta`.
    #[inline]
    fn translate(mut self, delta: Vec2) -> Self {
        match &mut self {
            Segment::Line(line) => {
                *line = line.translate(delta);
            }
            Segment::Quadratic(curve) => {
                *curve = curve.translate(delta);
            }
        }

        self
    }
}

/// A shape consisting of a number of segments.
#[derive(Debug, Default, Clone)]
struct Shape {
    segments: Vec<Segment>,
}

impl Shape {
    /// Get the bounding box of the shape.
    #[inline]
    fn bounding_box(&self) -> Rect {
        let mut abs_bb: Option<Rect> = None;

        for bb in self.segments.iter().map(Segment::bounding_box) {
            abs_bb.replace(abs_bb
                .map(|abs_bb| abs_bb.bounding_rect(bb))
                .unwrap_or(bb)
            );
        }

        abs_bb.unwrap_or_default()
    }

    /// Scale the shape by `scale`.
    #[inline]
    fn scale(&mut self, scale: f32) {
        for seg in &mut self.segments {
            *seg = seg.scale(scale);
        }
    }

    /// Translate the shape by `delta`.
    #[inline]
    fn translate(&mut self, delta: Vec2) {
        for seg in &mut self.segments {
            *seg = seg.translate(delta);
        }
    }
}


#[derive(Default)]
struct ShapeBuilder {
    pos: Vec2,
    shape: Shape,
}

impl ShapeBuilder {
    fn move_to(&mut self, pos: Vec2) {
        self.pos = pos;
    }

    fn line_to(&mut self, p1: Vec2) {
        let line = Line::new(self.pos, p1);

        self.pos = p1;
        self.shape.segments.push(Segment::Line(line));
    }

    fn quad_to(&mut self, p1: Vec2, p2: Vec2) {
        let curve = Quadratic::new(self.pos, p1, p2);

        self.pos = p2;
        self.shape.segments.push(Segment::Quadratic(curve));
    }
}

impl OutlineBuilder for ShapeBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.move_to(Vec2::new(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.line_to(Vec2::new(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.quad_to(Vec2::new(x1, y1), Vec2::new(x2, y2));
    }

    fn curve_to(&mut self, _x1: f32, _y1: f32, _x2: f32, _y2: f32, _x3: f32, _y3: f32) {
        todo!();
    }

    fn close(&mut self) {}
}

struct AtlasAllocator {
    free_rects: Vec<Rect>,
    used_rects: Vec<Rect>,
}

impl AtlasAllocator {
    fn new(width: f32, height: f32) -> Self {
        let used_rects = Vec::new();
        let free_rects = vec![
            Rect::from_corners(Vec2::ZERO, Vec2::new(width, height)),
        ];

        Self { free_rects, used_rects}
    }

    /// Allocate area with exactly the dimensions of `width` and `height`.
    fn alloc(&mut self, width: f32, height: f32) -> Option<Rect> {
        let index = self.free_rects.iter().position(|area| {
            area.width() >= width && area.height() >= height 
        })?;

        let area = self.free_rects.swap_remove(index);

        let (area, left_margin) = area.split_vertically(width);
        let (area, bottom_margin) = area.split_horizontally(height);

        if bottom_margin.width() > 0.0 && bottom_margin.height() > 0.0 {
            self.free_rects.push(bottom_margin);
        }

        if left_margin.width() > 0.0 && left_margin.height() > 0.0 {
            self.free_rects.push(left_margin);
        }

        self.used_rects.push(area);
        self.free_rects.sort_unstable_by_key(|a| {
            a.area() as u32
        });
        
        Some(area)
    }

    fn dimensions(&self) -> (u32, u32) {
        let mut bb = Rect::ZERO;

        for area in &self.used_rects {
            bb = bb.bounding_rect(*area);
        }

        (bb.max.x.ceil() as u32 + 1, bb.max.y.ceil() as u32 + 1)
    }
}

fn render_glyph(image: &mut image::GrayImage, info: GenerateInfo, glyph: &GlyphTemplate) {
    let atlas_bb = glyph.atlas_bb;

    let x_range = (atlas_bb.min.x.floor() as u32)..=(atlas_bb.max.x.ceil() as u32);
    let y_range = (atlas_bb.min.y.floor() as u32)..=(atlas_bb.max.y.ceil() as u32);

    let shadow_recip = info.shadow.recip();

    // Iterate through x and y in pixel space.
    for y in y_range.clone() {
        for x in x_range.clone() {
            let pixel = image.get_pixel_mut(x, y);

            // Point at the center of the pixel.
            let mut point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);

            // Translate the point from "atlas" space into "glyph" space and flip the y-coordinate.
            point -= atlas_bb.min;
            point.y = atlas_bb.height() - point.y;

            point.x -= info.left_padding;
            point.y -= info.top_padding;

            let mut distance = f32::MAX;
            let mut orthogonality = 0.0;

            for seg in &glyph.glyph_shape.shape.segments {
                let sd = seg.signed_dist(point);

                // If the distances are close, we use the orthogonality as a sort of tie breaker.
                // The more orthogonal, the better.
                let closer = if (sd.distance.abs() - distance.abs()).abs() <= 0.001 {
                    sd.orthogonality > orthogonality
                } else {
                    sd.distance.abs() < distance.abs()  
                };

                if closer {
                    orthogonality = sd.orthogonality;
                    distance = sd.distance;
                }
            }

            let sd = (-distance * shadow_recip).clamp(-1.0, 1.0) * 0.5 + 0.5;
            let sd = (sd * 255.0) as u8;

            *pixel = image::Luma([sd]);
        }
    }
}

pub(crate) struct GlyphShape {
    shape: Shape,
    codepoint: char,
}

pub(crate) fn load_glyph_shapes(data: &[u8], info: GenerateInfo) -> Result<Vec<GlyphShape>> {
    let face = Face::parse(data, 0)?;

    let shapes: Result<Vec<_>> = info.ranges
        .iter()
        .flat_map(|range| {
            range.clone().map(|codepoint| {
                let Some(index) = face.glyph_index(codepoint) else {
                    return Err(anyhow!("codepoint '{codepoint}' doesn't exist in font"));
                };

                let mut builder = ShapeBuilder::default();
                face.outline_glyph(index, &mut builder);

                // Scale the shapes. Then translate them such that their bounding box minimum lies
                // at zero.
                builder.shape.scale(info.scale);
                builder.shape.translate(Vec2::ZERO - builder.shape.bounding_box().min);

                Ok(GlyphShape { shape: builder.shape, codepoint })
            })
        })
        .collect();

    let mut shapes = shapes?;

    // Sort the shapes to largest area to smallest.
    shapes.sort_unstable_by_key(|glyph| {
        std::cmp::Reverse(glyph.shape.bounding_box().area() as u32)
    });

    Ok(shapes)
}

pub(crate) struct GlyphTemplate<'a> {
    glyph_shape: &'a GlyphShape,
    atlas_bb: Rect,
}

pub(crate) struct AtlasTemplate<'a> {
    glyphs: Vec<GlyphTemplate<'a>>,
    image: image::GrayImage,
}

impl<'a> AtlasTemplate<'a> {
    pub(crate) fn new(shapes: &'a [GlyphShape], info: GenerateInfo) -> Self {
        let mut dim = Vec2::new(info.atlas_width as f32, info.atlas_height as f32);
        let mut allocator = AtlasAllocator::new(dim.x, dim.y);

        let glyphs = loop {
            let glyphs: Option<Vec<_>> = shapes
                .iter()
                .map(|glyph_shape| {
                    let glyph_bb = glyph_shape.shape.bounding_box();

                    let width = glyph_bb.width()
                        + info.right_padding
                        + info.left_padding;

                    let height = glyph_bb.height()
                        + info.top_padding
                        + info.bottom_padding;

                    let atlas_bb = allocator.alloc(width.ceil(), height.ceil())?;

                    Some(GlyphTemplate { glyph_shape, atlas_bb })
                })
                .collect();

            let Some(glyphs) = glyphs else {
                dim *= 1.5;
                allocator = AtlasAllocator::new(dim.x, dim.y);

                continue;
            };

            break glyphs;
        };

        let (width, height) = allocator.dimensions();
        let image = image::GrayImage::new(width, height); 

        Self { glyphs, image }
    }

    pub(crate) fn render(mut self, info: GenerateInfo) -> image::GrayImage {
        for glyph in &self.glyphs {
            render_glyph(&mut self.image, info, glyph);
        }

        self.image
    }
}
