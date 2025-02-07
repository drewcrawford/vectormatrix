/**
A vector type.
*/
pub struct Vec<T, const N: usize>([T; N]);

impl <T, const N: usize> Vec<T, N> {
    /**
    Creates a new vector.
*/
    pub const fn new(value: [T; N]) -> Self {
        Self(value)
    }
}

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