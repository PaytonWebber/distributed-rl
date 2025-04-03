use serde::{Deserialize, Serialize};
use serde_json;
use tokio;
use zeromq::*;

#[derive(Debug, Deserialize, Serialize)]
struct Experience {
    state: Vec<f32>,
    policy: Vec<f32>,
    reward: f32,
}

async fn pull_experiences(mut pull_socket: PullSocket) {
    loop {
        let recv_message: String = match pull_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving pull message: {}", e);
                continue;
            }
            Ok(msg) => msg.try_into().unwrap(),
        };

        let experiences: Result<Vec<Experience>, _> = serde_json::from_str(&recv_message);
        match experiences {
            Ok(exps) => {
                println!("Received {} experiences:", exps.len());
                for e in exps {
                    println!("{:?}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to parse JSON: {}", e);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pull_socket = PullSocket::new();
    pull_socket.bind("tcp://0.0.0.0:5555").await?;

    let pull_task = pull_experiences(pull_socket);

    // join3(pull_task, pub_task, rep_task).await;
    pull_task.await;
    Ok(())
}
