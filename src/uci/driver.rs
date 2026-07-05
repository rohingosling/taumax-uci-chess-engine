//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   UCI command loop and command dispatch.
//
//----------------------------------------------------------------------------------------------------------------------

use std::io;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use crate::engine::configuration::OptionUpdateStatus;
use crate::engine::selector::MoveSelector;
use crate::engine::session::EngineSession;
use crate::uci::command::UciCommand;
use crate::uci::parser::parse_uci_command;
use crate::uci::response;

//----------------------------------------------------------------------------------------------------------------------
// Function: run_uci_driver
//
// Description:
//
//   Read UCI command lines, dispatch each command, and stop when quit is received.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn run_uci_driver<R, W, S>(
    input: R,
    mut output: W,
    session: &mut EngineSession<S>,
) -> io::Result<()>
where
    R: BufRead + Send + 'static,
    W: Write,
    S: MoveSelector,
{
    let command_receiver = spawn_input_reader(input, session.search_stop_flag());

    while let Ok(line_result) = command_receiver.recv() {

        // BufRead::lines returns Result<String, io::Error>. The question mark keeps the loop small:
        // if reading stdin fails, the error is returned to main immediately.

        let line = line_result?;

        // Parsing converts loose protocol text into an enum so the rest of the engine can use normal
        // Rust pattern matching instead of string comparisons everywhere.

        let command = parse_uci_command(&line);

        // dispatch_uci_command returns false only for "quit"; every other command keeps the session
        // alive so the GUI can continue the conversation.

        if !dispatch_uci_command(command, &mut output, session)? {
            break;
        }
    }

    Ok(())
}

//----------------------------------------------------------------------------------------------------------------------
// Function: spawn_input_reader
//
// Description:
//
//   Read UCI input on a helper thread and raise the stop flag as soon as a stop command is seen.
//
//----------------------------------------------------------------------------------------------------------------------

fn spawn_input_reader<R>(
    input: R,
    stop_requested: Arc<AtomicBool>,
) -> mpsc::Receiver<io::Result<String>>
where
    R: BufRead + Send + 'static,
{
    let (command_sender, command_receiver) = mpsc::channel();

    thread::spawn(move || {
        for line_result in input.lines() {
            if let Ok(line) = &line_result {
                if matches!(parse_uci_command(line), UciCommand::Stop) {
                    stop_requested.store(true, Ordering::Relaxed);
                }
            }

            if command_sender.send(line_result).is_err() {
                break;
            }
        }
    });

    command_receiver
}

//----------------------------------------------------------------------------------------------------------------------
// Function: dispatch_uci_command
//
// Description:
//
//   Execute one parsed UCI command and return whether the driver should continue.
//
//----------------------------------------------------------------------------------------------------------------------

fn dispatch_uci_command<W, S>(
    command: UciCommand,
    output: &mut W,
    session: &mut EngineSession<S>,
) -> io::Result<bool>
where
    W: Write,
    S: MoveSelector,
{
    match command {
        UciCommand::Uci => {

            // The UCI startup handshake is ordered: identity lines first, then "uciok" to announce
            // that the engine has finished describing itself.

            writeln!(output, "{}", response::engine_name())?;
            writeln!(output, "{}", response::engine_author())?;

            for option_line in response::engine_option_lines() {
                writeln!(output, "{option_line}")?;
            }

            writeln!(output, "{}", response::uciok())?;
            output.flush()?;
        }
        UciCommand::IsReady => {

            // Chess GUIs use "isready" as a synchronization point after setup commands.

            writeln!(output, "{}", response::readyok())?;
            output.flush()?;
        }
        UciCommand::UciNewGame => {

            // Reset only the per-game state. The process and configured selector remain alive.

            session.new_game();
        }
        UciCommand::Position(position_command) => {

            // Bad position text should not crash the engine process. UCI diagnostics belong on stderr
            // because stdout is machine-read protocol output.

            if let Err(error) = session.set_position(position_command) {
                eprintln!("position error: {error}");
            }
        }
        UciCommand::Go(limits) => {

            // Run the current search synchronously, obeying cooperative stop and deadline checks, then
            // print the required bestmove response shape.

            let search_result = session.search(&limits);

            for trace_line in search_result.trace_lines {
                writeln!(output, "{trace_line}")?;
            }

            writeln!(
                output,
                "{}",
                response::bestmove(&search_result.best_move_text)
            )?;
            output.flush()?;
        }
        UciCommand::Stop => {

            // The input reader raises the stop flag immediately. By the time dispatch reaches this
            // command, a synchronous search has already observed it and returned.

            session.clear_search_stop();
        }
        UciCommand::Debug(debug_enabled) => {
            session.set_debug_enabled(debug_enabled);
        }
        UciCommand::SetOption { name, value } => {

            // Known options are validated and stored in the session. Unknown GUI options are ignored
            // for compatibility with front ends that send common settings such as Hash or Threads.

            match session.set_option(&name, value.as_deref()) {
                Ok(OptionUpdateStatus::Updated) => {}
                Ok(OptionUpdateStatus::Unknown) => {
                    if session.is_debug_enabled() {
                        eprintln!("debug: ignored option '{name}' with value '{value:?}'");
                    }
                }
                Err(error) => {
                    eprintln!("option error: {error}");
                }
            }
        }
        UciCommand::Quit => {
            return Ok(false);
        }
        UciCommand::Unknown(text) => {

            // The protocol allows engines to ignore commands they do not understand.

            if session.is_debug_enabled() {
                eprintln!("debug: ignored unknown command '{text}'");
            }
        }
    }

    Ok(true)
}
