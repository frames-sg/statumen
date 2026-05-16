use std::process::ExitCode;

mod commands;

fn main() -> ExitCode {
    match commands::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask failed: {err}");
            ExitCode::FAILURE
        }
    }
}
