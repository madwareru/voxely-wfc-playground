use std::f32::consts::PI;
use std::time::Instant;
use parry3d::math::{Point, Vector, Real};
use parry3d::query::Ray;
use rand::{Rng, thread_rng};

const GOLDEN_RATIO: f32 = PI * (3.2360679775);

pub struct StopWatch {
    instant: Instant,
    name: &'static str
}

impl StopWatch {
    pub fn named(name: &'static str) -> Self {
        Self { name, instant: Instant::now() }
    }
}

impl Drop for StopWatch {
    fn drop(&mut self) {
        println!("{}: {} ms", self.name, self.instant.elapsed().as_secs_f32() * 1000.0)
    }
}

pub fn sun_flower_direction_on_a_unit_sphere(range: usize, sample: usize) -> Vector<Real> {
    let sample = sample as f32 + 0.5;
    let phi_cos = 1.0 - 2.0 * sample / range as f32;
    let theta = GOLDEN_RATIO * sample;
    let phi_sin = (1.0 - phi_cos * phi_cos).sqrt();
    [phi_sin * theta.cos(), phi_sin * theta.sin(), phi_cos].into()
}

pub fn random_direction_on_a_unit_sphere() -> Vector<Real> {
    let mut rng = thread_rng();
    sun_flower_direction_on_a_unit_sphere(0x400000, rng.gen_range(0..0x400000)).normalize()
}
