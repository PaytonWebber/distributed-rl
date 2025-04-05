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
    let request_message: String = "REQUEST_BATCH".to_string();
    let update_message: String = "SENDING_PARAMETERS".to_string();
    loop {
        // receive (MESSAGE)
        let recv_message: String = match rep_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving from pull socket {}", e);
                continue;
            }
            Ok(msg) => {
                let raw: String = msg.try_into().unwrap();
                raw.trim_end_matches('\0').to_string()
            }
        };
        if recv_message == request_message {
            let buffer = replay_buffer.lock().await;

            if buffer.len() < 1024 {
                // send (ACK)
                if let Err(e) = rep_socket.send("NO".into()).await {
                    eprintln!("Failed to acknowledge parameter update: {}", e);
                }
                continue;
            }

            // send (ACK)
            if let Err(e) = rep_socket.send("OK".into()).await {
                eprintln!("Failed to acknowledge parameter update: {}", e);
            }

            // receive (ACK)
            let _ack = match rep_socket.recv().await {
                Err(e) => {
                    eprintln!("Error receiving from pull socket {}", e);
                    continue;
                }
                Ok(msg) => msg,
            };

            let batch = buffer.sample(32);
            drop(buffer);

            // send (BATCH)
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
        } else if recv_message == update_message {
            // send (ACK)
            if let Err(e) = rep_socket.send("OK".into()).await {
                eprintln!("Failed to acknowledge parameter update: {}", e);
            }

            // receive (PARAMETERS)
            let _parameters_bytes: Vec<u8> = match rep_socket.recv().await {
                Err(e) => {
                    eprintln!("Error receiving from pull socket {}", e);
                    continue;
                }
                Ok(msg) => msg.try_into().unwrap(),
            };
            // send (ACK)
            if let Err(e) = rep_socket.send("OK".into()).await {
                eprintln!("Failed to acknowledge parameter update: {}", e);
            }
        }
    }
}
