mod generate;

use anyhow::Result;

use std::collections::HashMap;
use std::ops::RangeInclusive;

use generate::AtlasTemplate;
use rendi_asset::{Image, ImageFormat, RawFormat};
use rendi_math::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct GenerateInfo<'a> {
    /// The scale the shapes will be rendered at in terms of pixels per font unit.
    ///
    /// Note that this doesn't effect the glyph data, which will not be scaled.
    pub scale: f32,

    /// The width of the shadow in pixels.
    ///
    /// This indicates the width of the the distance area around the glyph, which in turn
    /// affects the distance falloff rate.
    pub shadow: f32,

    /// The target width of the atlas. It may be bigger if the glyphs doesn't fit.
    pub atlas_width: u32,

    /// The target height of the atlas. It may be bigger if the glyphs doesn't fit.
    pub atlas_height: u32,

    /// The character ranges to render to the atlas.
    pub ranges: &'a [RangeInclusive<char>],
}

impl<'a> Default for GenerateInfo<'a> {
    fn default() -> Self {
        Self {
            scale: 0.1,
            shadow: 12.0,
            atlas_width: 1000,
            atlas_height: 1000,
            ranges: &['\u{20}'..='\u{7e}'],
        }
    }
}

/// Data relating to a glyph.
///
/// All units are directly from the font and not scaled.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Glyph {
    atlas_rect: Rect,
    advance: f32,
    dim: Vec2,
    offset: Vec2,
}

impl Glyph {
    /// Bounding rectangle on the atlas containing the glyph.
    ///
    /// Coordinate range `0.0` to `1.0`.
    pub fn atlas_rect(&self) -> Rect {
        self.atlas_rect
    }

    /// Get the horizontal advance of the glyph.
    pub fn advance(&self) -> f32 {
        self.advance
    }

    /// Get the dimension of the glyph.
    pub fn dim(&self) -> Vec2 {
        self.dim
    }

    /// Get the offset from the top and left where the glyph should be drawn.
    ///
    /// This accounts for the offset the glpyhs are stored in the atlas compared to how they are
    /// defined.
    pub fn offset(&self) -> Vec2 {
        self.offset
    }
}

/// A signed distance field glyph atlas.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Atlas {
    glyphs: HashMap<char, Glyph>,
    kernings: HashMap<(char, char), f32>,
    image: Image,
}

impl Atlas {
    pub fn new(ttf: &[u8], info: GenerateInfo) -> Result<Self> {
        let face = ttf_parser::Face::parse(ttf, 0)?;
        let font_size = info.scale.recip();

        let glyph_shapes = generate::load_glyph_shapes(&face, info)?;
        let mut template = AtlasTemplate::new(&glyph_shapes, info);

        template.render(info);

        let (atlas_width, atlas_height) = template.image.dimensions();
        let image_data = template.image.into_raw();

        let glyphs = template.glyphs
            .iter()
            .map(|(atlas_bb, glyph)| {
                let bb = glyph.shape.bounding_box();
                let shadow_width = info.shadow * font_size;

                let dim = Vec2::new(bb.width(), bb.height());
                let advance = face.glyph_hor_advance(glyph.id).unwrap() as f32;

                let dim = dim + shadow_width * 2.0;
                let mut offset = glyph.offset;
                offset.x *= -1.0;

                let atlas_size = UVec2::new(atlas_width, atlas_height).as_vec2();
                let atlas_rect = Rect {
                    min: atlas_bb.min / atlas_size,
                    max: atlas_bb.max / atlas_size,
                };

                (glyph.codepoint, Glyph {
                    atlas_rect,
                    advance,
                    offset,
                    dim,
                })
            })
            .collect();

        let kernings = face
            .tables()
            .kern
            .map(|kern| kern.subtables
                .into_iter()
                .next()
                .map(|subtable| {
                    let mut pairs = HashMap::new();

                    for (_, a) in template.glyphs.iter() {
                        for (_, b) in template.glyphs.iter() {
                            let kerning = subtable.glyphs_kerning(a.id, b.id);
                            
                            if let Some(kerning) = kerning {
                                pairs.insert((a.codepoint, b.codepoint), kerning as f32);
                            }
                        }
                    }

                    pairs
                })
            )
            .flatten()
            .unwrap_or_default();

        let format = ImageFormat::Raw(RawFormat::R8Unorm);
        let image = Image {
            mips: vec![image_data],
            width: atlas_width,
            height: atlas_height,
            format,
        };

        Ok(Self { image, glyphs, kernings })
    }

    /// Get the signed distance field glyph atlas image.
    pub fn image(&self) -> &Image {
        &self.image
    }

    /// Get kerning offset for two chars `a` and `b`.
    pub fn kerning(&self, a: char, b: char) -> f32 {
        self.kernings.get(&(a, b)).copied().unwrap_or(0.0)
    }

    /// Get the [`Glyph`] for `codepoint`.
    pub fn glyph(&self, codepoint: char) -> Option<Glyph> {
        self.glyphs.get(&codepoint).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_font() {
        let bytes = include_bytes!("../font.ttf");

        let atlas = Atlas::new(
            bytes,
            GenerateInfo {
                atlas_width: 1000,
                atlas_height: 1000,

                shadow: 12.0,
                scale: 0.1,

                ranges: &['i'..='i'],
            },
        ).unwrap();

        let image = image::GrayImage::from_raw(
            atlas.image.width,
            atlas.image.height,
            atlas.image.mips[0].clone(),
        );

        image.unwrap().save("font.png").unwrap();
    }
}
