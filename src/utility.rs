use std::f32::consts::PI;
use std::time::Instant;
use parry3d::math::{Point, Vector, Real};
use parry3d::query::Ray;
use rand::{Rng, thread_rng};

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
    let phi = phi_cos.acos();
    let theta = PI * (1.0 + 5.0f32.sqrt()) * sample;
    let phi_sin = phi.sin();
    [phi_sin * theta.cos(), phi_sin * theta.sin(), phi_cos].into()
}

pub fn sun_flower_ray_on_a_hemisphere(
    range: usize,
    sample: usize,
    origin: Point<Real>,
    direction: Vector<Real>
) -> Ray {
    let dir: Vector<Real> = sun_flower_direction_on_a_unit_sphere(range, sample);
    if dir.dot(&direction) >= 0.0 {
        Ray::new(origin, dir)
    } else {
        Ray::new(origin, -dir)
    }
}

pub fn random_dir_on_a_unit_sphere() -> Vector<Real> {
    let mut rng = thread_rng();
    let xi: [f32; 2] = rng.gen();
    [
        xi[0].sqrt() * (xi[1] * PI * 2.0).cos(),
        xi[0].sqrt() * (xi[1] * PI * 2.0).sin(),
        (1.0 - xi[0]).cos()
    ].into()
}

pub fn random_direction_on_a_unit_sphere() -> Vector<Real> {
    let mut rng = thread_rng();
    sun_flower_direction_on_a_unit_sphere(0x400000, rng.gen_range(0..0x400000)).normalize()
}

pub fn random_ray_on_a_hemisphere(origin: Point<Real>, direction: Vector<Real>) -> Ray {
    let dir: Vector<Real> = random_direction_on_a_unit_sphere();
    if dir.dot(&direction) >= 0.0 {
        Ray::new(origin, dir)
    } else {
        Ray::new(origin, -dir)
    }
}

pub fn random_ray_on_a_hemisphere2(origin: Point<Real>, direction: Vector<Real>) -> Ray {
    let dir: Vector<Real> = random_dir_on_a_unit_sphere();
    if dir.dot(&direction) >= 0.0 {
        Ray::new(origin, dir)
    } else {
        Ray::new(origin, -dir)
    }
}