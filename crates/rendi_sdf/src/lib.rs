mod generate;

use anyhow::Result;

use std::collections::HashMap;
use std::ops::RangeInclusive;

use rendi_math::prelude::*;
use generate::AtlasTemplate;

#[derive(Debug, Clone, Copy)]
pub struct GenerateInfo<'a> {
    /// The scale the shapes will be renderer at.
    pub scale: f32,

    /// The width of the shadow.
    pub shadow: f32,

    /// The padding width on the left of each shape.
    pub left_padding: f32,

    /// The padding width on the right of each shape.
    pub right_padding: f32,

    /// The padding height on top of each shape.
    pub top_padding: f32,

    /// The padding height on the bottom of each shape.
    pub bottom_padding: f32,

    /// The target width of the atlas. It may be bigger if the glyphs doesn't fit.
    pub atlas_width: u32,

    /// The target height of the atlas. It may be bigger if the glyphs doesn't fit.
    pub atlas_height: u32,

    /// The character ranges to render to the atlas.
    pub ranges: &'a [RangeInclusive<char>],
}

#[derive(Debug, Clone, Copy)]
pub struct Glyph {
    atlas_rect: Rect,
    ver_advance: f32,
    hor_advance: f32,
}

impl Glyph {
    /// Bounding rectangle on the atlas containing the glyph.
    ///
    /// Coordinate range `0.0` to `1.0`.
    pub fn atlas_rect(&self) -> Rect {
        self.atlas_rect
    }

    pub fn ver_advance(&self) -> f32 {
        self.ver_advance
    }

    pub fn hor_advance(&self) -> f32 {
        self.hor_advance
    }
}

#[derive(Debug, Clone)]
pub struct Atlas {
    #[allow(dead_code)]
    glyphs: HashMap<char, Glyph>,

    image: image::GrayImage,
}

impl Atlas {
    pub fn new(ttf: &[u8], info: GenerateInfo) -> Result<Self> {
        let glyph_shapes = generate::load_glyph_shapes(ttf, info)?;
        let image = AtlasTemplate::new(&glyph_shapes, info).render(info);

        Ok(Self { image, glyphs: HashMap::new() })
    }

    pub fn image(&self) -> &image::GrayImage {
        &self.image
    }

    pub fn image_data(&self) -> &[u8] {
        &self.image
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font() {
        let bytes = include_bytes!("../font.ttf"); 

        let atlas = Atlas::new(bytes, GenerateInfo {
            bottom_padding: 12.0,
            top_padding: 12.0,
            right_padding: 12.0,
            left_padding: 12.0,
           
            atlas_width: 1000,
            atlas_height: 1000,

            shadow: 12.0,
            scale: 0.1,

            ranges: &['a'..='z'],
        });

        atlas.unwrap().image.save("font.png").unwrap();
    }
}
