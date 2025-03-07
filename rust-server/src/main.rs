use futures::future::join3;
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use tokio;
use tokio::sync::Mutex;
use zeromq::*;

#[derive(Debug, Deserialize, Serialize)]
struct Experience {
    model_hash: String,
    states: Vec<Vec<f64>>,
    policies: Vec<Vec<f64>>,
    rewards: Vec<Vec<f64>>,
}

async fn run_pull_task(mut pull_socket: PullSocket, replay_buffer: Arc<Mutex<Vec<Experience>>>) {
    loop {
        let recv_message: String = match pull_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving pull message: {}", e);
                continue;
            }
            Ok(msg) => msg.try_into().unwrap(),
        };

        let experience: Experience = match serde_json::from_str(&recv_message) {
            Ok(exp) => exp,
            Err(e) => {
                eprintln!("Direct parsing failed: {}", e);
                continue;
            }
        };

        let mut buffer = replay_buffer.lock().await;
        buffer.push(experience);
        println!("Updated replay buffer: {:?}", *buffer);
    }
}

async fn run_pub_task(mut pub_socket: PubSocket) {
    let mut count = 0;
    loop {
        let msg = ZmqMessage::from(format!("Broadcast message {} from PUB", count));
        if let Err(e) = pub_socket.send(msg).await {
            eprintln!("Error sending on PUB socket: {:?}", e);
        } else {
            println!("PUB socket sent a broadcast message");
        }
        count += 1;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

async fn run_rep_task(mut rep_socket: RepSocket, replay_buffer: Arc<Mutex<Vec<Experience>>>) {
    loop {
        let _recv_message: String = match rep_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving on REP socket: {}", e);
                continue;
            }
            Ok(msg) => msg.try_into().unwrap(),
        };

        let mut buffer = replay_buffer.lock().await;
        if buffer.len() < 4 {
            let reply = ZmqMessage::from("NOT ENOUGH EXPERIENCES");
            if let Err(e) = rep_socket.send(reply).await {
                eprintln!("Error sending reply on REP socket: {:?}", e);
            }
            continue;
        }
        let experience = buffer.pop().unwrap();
        drop(buffer);

        let send_json = match serde_json::to_string(&experience) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("Error serializing experience: {}", e);
                continue;
            }
        };

        let send_msg = ZmqMessage::from(send_json);
        if let Err(e) = rep_socket.send(send_msg).await {
            eprintln!("Error sending reply on REP socket: {:?}", e);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pull_socket = PullSocket::new();
    pull_socket.bind("tcp://0.0.0.0:5555").await?;

    let mut pub_socket = PubSocket::new();
    pub_socket.bind("tcp://0.0.0.0:5556").await?;

    let mut rep_socket = RepSocket::new();
    rep_socket.bind("tcp://0.0.0.0:5557").await?;

    let replay_buffer = Arc::new(Mutex::new(Vec::<Experience>::new()));

    let pull_buffer = Arc::clone(&replay_buffer);
    let rep_buffer = Arc::clone(&replay_buffer);

    let pull_task = run_pull_task(pull_socket, pull_buffer);
    let pub_task = run_pub_task(pub_socket);
    let rep_task = run_rep_task(rep_socket, rep_buffer);

    join3(pull_task, pub_task, rep_task).await;
    Ok(())
}
