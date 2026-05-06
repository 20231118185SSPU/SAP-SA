//! Standalone binary that exports TypeScript type definitions from Rust protocol types.
//!
//! Usage: `cargo run -p sa --bin export_types`
//!
//! This generates `.ts` files in `crates/bindings/` that mirror the Rust types
//! in `sa_core::ws_protocol`. Run this whenever the protocol types change.

use sa_core::ws_protocol::*;
use ts_rs::TS;

fn main() {
    // Client messages
    ClientMessage::export().expect("failed to export ClientMessage");
    eprintln!("✓ ClientMessage.ts");

    // Server messages
    ServerMessage::export().expect("failed to export ServerMessage");
    eprintln!("✓ ServerMessage.ts");

    // Individual types
    MemoryFact::export().expect("failed to export MemoryFact");
    eprintln!("✓ MemoryFact.ts");

    UserQuestion::export().expect("failed to export UserQuestion");
    eprintln!("✓ UserQuestion.ts");

    QuestionOption::export().expect("failed to export QuestionOption");
    eprintln!("✓ QuestionOption.ts");

    QuestionMode::export().expect("failed to export QuestionMode");
    eprintln!("✓ QuestionMode.ts");

    UserQuestionAnswer::export().expect("failed to export UserQuestionAnswer");
    eprintln!("✓ UserQuestionAnswer.ts");

    Event::export().expect("failed to export Event");
    eprintln!("✓ Event.ts");

    AgentIdentity::export().expect("failed to export AgentIdentity");
    eprintln!("✓ AgentIdentity.ts");

    // Init types
    InitMethod::export().expect("failed to export InitMethod");
    eprintln!("✓ InitMethod.ts");

    InitMethodOption::export().expect("failed to export InitMethodOption");
    eprintln!("✓ InitMethodOption.ts");

    InitRequired::export().expect("failed to export InitRequired");
    eprintln!("✓ InitRequired.ts");

    InitCompleted::export().expect("failed to export InitCompleted");
    eprintln!("✓ InitCompleted.ts");

    InitFailed::export().expect("failed to export InitFailed");
    eprintln!("✓ InitFailed.ts");

    // Connection types
    ClientHello::export().expect("failed to export ClientHello");
    eprintln!("✓ ClientHello.ts");

    ServerHello::export().expect("failed to export ServerHello");
    eprintln!("✓ ServerHello.ts");

    HelloReject::export().expect("failed to export HelloReject");
    eprintln!("✓ HelloReject.ts");

    // OpenAI types
    sa_core::openai::WireApi::export().expect("failed to export WireApi");
    eprintln!("✓ WireApi.ts");

    sa_core::openai::AuthStyle::export().expect("failed to export AuthStyle");
    eprintln!("✓ AuthStyle.ts");

    eprintln!("\nAll types exported to crates/bindings/");
}
