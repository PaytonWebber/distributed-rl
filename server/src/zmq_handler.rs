use crate::experience::Experience;
use crate::replay_buffer::ReplayBuffer;
use serde_json;
use std::sync::Arc;
use tokio::sync::Mutex;
use zeromq::*;

pub async fn pull_experiences(
    mut pull_socket: PullSocket,
    replay_buffer: Arc<Mutex<ReplayBuffer>>,
) {
    loop {
        let recv_message: String = match pull_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving from pull socket {}", e);
                continue;
            }
            Ok(msg) => msg.try_into().unwrap(),
        };

        let experiences: Result<Vec<Experience>, _> = serde_json::from_str(&recv_message);
        match experiences {
            Ok(exps) => {
                println!("Received {} experiences:", exps.len());
                let mut buffer = replay_buffer.lock().await;
                for exp in exps {
                    buffer.push(exp);
                }
            }
            Err(e) => eprintln!("Failed to parse JSON: {}", e),
        }
    }
}

pub async fn rep_task(mut rep_socket: RepSocket, replay_buffer: Arc<Mutex<ReplayBuffer>>) {
    loop {
        let recv_message: String = match rep_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving from pull socket {}", e);
                continue;
            }
            Ok(msg) => msg.try_into().unwrap(),
        };
        println!("Received: {}", recv_message);

        println!("Sending batch");
        let buffer = replay_buffer.lock().await;
        let batch = buffer.sample(32);
        drop(buffer);

        match serde_json::to_string(&batch) {
            Ok(msg) => {
                if let Err(e) = rep_socket.send(msg.into()).await {
                    eprintln!("Failed to send mini-batch: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to serialize batch: {}", e);
            }
        }
    }
}
