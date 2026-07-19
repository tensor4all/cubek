use std::{
    error::Error,
    fmt::{Debug, Display},
};

/// Error that arises from invalid configurations
pub struct InvalidConfigError {
    diagnostic: Box<dyn InvalidConfigDiagnostic>,
}

trait InvalidConfigDiagnostic: Debug + Display + Send + Sync + 'static {}

impl<T> InvalidConfigDiagnostic for T where T: Debug + Display + Send + Sync + 'static {}

impl InvalidConfigError {
    /// Wrap a diagnostic value in a thread-safe configuration error.
    pub fn new<T>(diagnostic: T) -> Self
    where
        T: Debug + Display + Send + Sync + 'static,
    {
        Self {
            diagnostic: Box::new(diagnostic),
        }
    }
}

impl Debug for InvalidConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.diagnostic, f)
    }
}

impl Display for InvalidConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.diagnostic, f)
    }
}

impl Error for InvalidConfigError {}

/// Error that arises from invalid configurations
pub struct FormattedConfigError {
    func: Box<dyn Fn() -> String + Send + Sync>,
}

impl Debug for FormattedConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FormattedConfigError")
            .field(&(self.func)())
            .finish()
    }
}

impl FormattedConfigError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<F: Fn() -> String + Send + Sync + 'static>(func: F) -> InvalidConfigError {
        InvalidConfigError::new(Self {
            func: Box::new(func),
        })
    }
}

impl Display for FormattedConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_error_send_sync<T: std::error::Error + Send + Sync>() {}

    #[test]
    fn invalid_config_error_is_a_thread_safe_error() {
        assert_error_send_sync::<InvalidConfigError>();
    }
}
