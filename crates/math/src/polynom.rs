use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub enum Roots {
    None,
    One([f32; 1]),
    Two([f32; 2]),
    Three([f32; 3]),
}

impl AsRef<[f32]> for Roots {
    fn as_ref(&self) -> &[f32] {
        match self {
            Roots::None => &[],
            Roots::One(roots) => roots,
            Roots::Two(roots) => roots,
            Roots::Three(roots) => roots,
        }
    }
}

pub fn linear_roots(a: f32, b: f32) -> Roots {
    if a == 0.0 {
        if b == 0.0 {
            Roots::One([0.0])
        } else {
            Roots::None
        }
    } else {
        Roots::One([-b / a])
    }
}

pub fn quadratic_roots(a: f32, b: f32, c: f32) -> Roots {
    if a == 0.0 {
        linear_roots(b, c)
    } else {
        let disc = b * b - 4.0 * a * c;

        if disc == 0.0 {
            Roots::One([-0.5 * b / a])
        } else if disc < 0.0 {
            Roots::None
        } else {
            let sqrt_disc = disc.sqrt();

            let r1 = (-b + sqrt_disc) / (2.0 * a);
            let r2 = (-b - sqrt_disc) / (2.0 * a);
             
            Roots::Two([r1, r2])
        }
    }
}

pub fn cubic_roots(a: f32, b: f32, c: f32, d: f32) -> Roots {
    if a == 0.0 {
        quadratic_roots(b, c, d)
    } else {
        let b = b / a;
        let c = c / a;
        let d = d / a;

        let q = (b * b - 3.0 * c) / 9.0;
        let r = (2.0 * b.powi(3) - 9.0 * b * c + 27.0 * d) / 54.0;

        let disc = q.powi(3) - r * r;

        if disc >= 0.0 {
            let theta = (r / q.powi(3).sqrt()).acos();
            let fac = -2.0 * q.sqrt();
            let add = -b * (1.0 / 3.0);

            Roots::Three([
                fac * ((1.0 / 3.0) * theta).cos() + add,
                fac * ((1.0 / 3.0) * (theta + PI * 2.0)).cos() + add,
                fac * ((1.0 / 3.0) * (theta + PI * 2.0 + PI * 2.0)).cos() + add,
            ])
        } else {
            let temp = ((-disc).sqrt() + r.abs()).powf(1.0 / 3.0);
            let sign = r.signum();

            Roots::One([
                -sign * (temp + q / temp) - (1.0 / 3.0) * b,
            ])
        }
    }
}
