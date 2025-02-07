use crate::types::Constants;

/**
A vector type.
*/
#[derive(Copy,Clone)]
pub struct Vec<T, const N: usize>([T; N]);

impl <T, const N: usize> Vec<T, N> {
    /**
    Creates a new vector.
*/
    pub const fn new(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T: Constants, const N: usize> Vec<T, N> {
    /**
    Creates a new vector with all components set to zero.
*/
    #[inline] pub const fn zero() -> Self {
        Self([T::ZERO; N])
    }

    /**
    Creates a new vector with all components set to one.
*/
    #[inline] pub const fn one() -> Self {
        Self([T::ONE; N])
    }
}

//getters

mod private {
    pub trait AtLeastOne {}
    impl<T> AtLeastOne for [T; 1] {}
    impl<T> AtLeastOne for [T; 2] {}
    impl<T> AtLeastOne for [T; 3] {}
    impl<T> AtLeastOne for [T; 4] {}
    pub trait AtLeastTwo{}
    impl<T> AtLeastTwo for [T; 2] {}
    impl<T> AtLeastTwo for [T; 3] {}
    impl<T> AtLeastTwo for [T; 4] {}

    pub trait AtLeastThree{}
    impl<T> AtLeastThree for [T; 3] {}
    impl<T> AtLeastThree for [T; 4] {}

    pub trait AtLeastFour{}
    impl<T> AtLeastFour for [T; 4] {}

}
impl<T, const N: usize> Vec<T, N>
where
    [T; N]: private::AtLeastOne
{
    /**
    Gets the first component.
*/
    #[inline] pub const fn x(&self) -> &T {
        &self.0[0]
    }
    /**
    Gets the first component.
    */
    #[inline] pub const fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }
}
impl<T, const N: usize> Vec<T, N>
where
    [T; N]: private::AtLeastTwo
{
    /**
    Gets the second component.
*/
    #[inline] pub const fn y(&self) -> &T {
        &self.0[1]
    }
    /**
    Gets the second component.
*/
    #[inline] pub const fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }
}

impl<T, const N: usize> Vec<T, N>
where
    [T; N]: private::AtLeastThree
{
    /**
    Gets the third component.
*/
    #[inline] pub const fn z(&self) -> &T {
        &self.0[2]
    }
    /**
    Gets the third component.
*/
    #[inline] pub const fn z_mut(&mut self) -> &mut T {
        &mut self.0[2]
    }
}

impl<T, const N: usize> Vec<T, N>
where
    [T; N]: private::AtLeastFour
{
    /**
    Gets the fourth component.
*/
    #[inline] pub const fn w(&self) -> &T {
        &self.0[3]
    }
    /**
    Gets the fourth component.
*/
    #[inline] pub const fn w_mut(&mut self) -> &mut T {
        &mut self.0[3]
    }
}

//elementwise ops

impl<T: core::ops::Add<Output=T> + Clone, const N: usize> Vec<T, N>
{
    #[inline] pub fn elementwise_add(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() + other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Add<Output=T> + Clone, const N: usize>  core::ops::Add for Vec<T, N> {
    type Output = Self;
    #[inline] fn add(self, other: Self) -> Self {
        self.elementwise_add(other)
    }
}

impl<T: core::ops::Sub<Output=T> + Clone, const N: usize> Vec<T, N>
{
    #[inline] pub fn elementwise_sub(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() - other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Sub<Output=T> + Clone, const N: usize>  core::ops::Sub for Vec<T, N> {
    type Output = Self;
    #[inline] fn sub(self, other: Self) -> Self {
        self.elementwise_sub(other)
    }
}

impl<T: core::ops::Mul<Output=T> + Clone, const N: usize> Vec<T, N> {
    #[inline] pub fn elementwise_mul(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() * other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Mul<Output=T> + Clone, const N: usize>  core::ops::Mul for Vec<T, N> {
    type Output = Self;
    #[inline] fn mul(self, other: Self) -> Self {
        self.elementwise_mul(other)
    }
}

impl<T: core::ops::Div<Output=T> + Clone, const N: usize> Vec<T, N> {
    #[inline] pub fn elementwise_div(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() / other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Div<Output=T> + Clone, const N: usize>  core::ops::Div for Vec<T, N> {
    type Output = Self;
    #[inline] fn div(self, other: Self) -> Self {
        self.elementwise_div(other)
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize>  Vec<T, N> {
    #[inline] fn add_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] += other.0[i].clone();
        }
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize>  core::ops::AddAssign for Vec<T, N> {
    #[inline] fn add_assign(&mut self, other: Self) {
        self.add_elementwise_assign(other)
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize>  Vec<T, N> {
    #[inline] fn sub_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] -= other.0[i].clone();
        }
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize>  core::ops::SubAssign for Vec<T, N> {
    #[inline] fn sub_assign(&mut self, other: Self) {
        self.sub_elementwise_assign(other)
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize>  Vec<T, N> {
    #[inline] fn mul_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] *= other.0[i].clone();
        }
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize>  core::ops::MulAssign for Vec<T, N> {
    #[inline] fn mul_assign(&mut self, other: Self) {
        self.mul_elementwise_assign(other)
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize>  Vec<T, N> {
    #[inline] fn div_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] /= other.0[i].clone();
        }
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize>  core::ops::DivAssign for Vec<T, N> {
    #[inline] fn div_assign(&mut self, other: Self) {
        self.div_elementwise_assign(other)
    }
}




#[cfg(test)] mod tests {
    use crate::vec::Vec;

    #[test] fn elementwise() {
        let a = Vec::new([1,2,3]);
        let b = Vec::new([4,5,6]);
        let c = a + b;
        assert_eq!(c.0, [5,7,9]);

        let d = a - b;
        assert_eq!(d.0, [-3,-3,-3]);

        let e = a * b;
        assert_eq!(e.0, [4,10,18]);

        let f = a / b;
        assert_eq!(f.0, [0,0,0]);
    }

    #[test] fn elementwise_assign() {
        let mut a = Vec::new([1,2,3]);
        let b = Vec::new([4,5,6]);
        a += b;
        assert_eq!(a.0, [5,7,9]);

        let mut c = Vec::new([1,2,3]);
        let d = Vec::new([4,5,6]);
        c -= d;
        assert_eq!(c.0, [-3,-3,-3]);

        let mut e = Vec::new([1,2,3]);
        let f = Vec::new([4,5,6]);
        e *= f;
        assert_eq!(e.0, [4,10,18]);

        let mut g = Vec::new([1,2,3]);
        let h = Vec::new([4,5,6]);
        g /= h;
        assert_eq!(g.0, [0,0,0]);
    }
}