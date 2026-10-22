#[tokio::main]
async fn main() {
    if let Err(error) = rustmoe_server::cli::run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
