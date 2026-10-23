use clap::Parser;
use rustmoe_server::agent::AgentConfig;
use rustmoe_server::agent_server::{serve, HttpAgentConfig};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "rustmoe-agent",
    about = "RustMoE Dev Agent powered by an OpenAI-compatible upstream (Strata)"
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8081")]
    addr: SocketAddr,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    upstream: String,
    #[arg(long, default_value = "qwen3.8-flash-next-iq3_s")]
    model: String,
    #[arg(long, default_value = "workspace")]
    agent_root: PathBuf,
    #[arg(long, default_value = "skills")]
    skills_dir: PathBuf,
    #[arg(long)]
    agent_allow_shell: bool,
    #[arg(long, default_value_t = 24)]
    agent_max_steps: usize,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    upstream_api_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = AgentConfig {
        root: args.agent_root,
        skills_dir: args.skills_dir,
        allow_shell: args.agent_allow_shell,
        max_steps: args.agent_max_steps,
    };
    serve(HttpAgentConfig {
        addr: args.addr,
        config,
        upstream: args.upstream,
        model: args.model,
        upstream_api_key: args.upstream_api_key,
        api_key: args.api_key,
    })
    .await
}
