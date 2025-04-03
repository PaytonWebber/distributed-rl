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

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ModelUpdate {
    model_hash: String,
    weights: Vec<f64>,
}

async fn pull_experiences(
    mut pull_socket: PullSocket,
    replay_buffer: Arc<Mutex<Vec<Experience>>>,
    latest_model: Arc<Mutex<Option<ModelUpdate>>>,
) {
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

        let lm = latest_model.lock().await;
        if let Some(model) = &*lm {
            if experience.model_hash != model.model_hash {
                println!("Experience does not use latest model. Dropping ...");
                continue;
            }
        }
        drop(lm);

        let mut buffer = replay_buffer.lock().await;
        buffer.push(experience);
        println!("Updated replay buffer: {:?}", *buffer);
    }
}

async fn publish_weights(mut pub_socket: PubSocket, latest_model: Arc<Mutex<Option<ModelUpdate>>>) {
    loop {
        {
            let lm = latest_model.lock().await;
            if let Some(model) = &*lm {
                if let Ok(json) = serde_json::to_string(model) {
                    let msg = ZmqMessage::from(json);
                    if let Err(e) = pub_socket.send(msg).await {
                        eprintln!("Error sending model update on PUB socket: {:?}", e);
                    } else {
                        println!("Published model update: {:?}", model);
                    }
                }
            } else {
                println!("No model update available yet.");
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

async fn run_rep_task(
    mut rep_socket: RepSocket,
    replay_buffer: Arc<Mutex<Vec<Experience>>>,
    latest_model: Arc<Mutex<Option<ModelUpdate>>>,
) {
    loop {
        let recv_message: String = match rep_socket.recv().await {
            Err(e) => {
                eprintln!("Error receiving on REP socket: {}", e);
                continue;
            }
            Ok(msg) => msg.try_into().unwrap(),
        };

        // Try to serialized it
        if let Ok(model_update) = serde_json::from_str::<ModelUpdate>(&recv_message) {
            {
                let mut lm = latest_model.lock().await;
                *lm = Some(model_update.clone());
            }
            let reply = ZmqMessage::from("Model update received");
            if let Err(e) = rep_socket.send(reply).await {
                eprintln!("Error sending model update ack: {:?}", e);
            }
            continue;
        }

        // Otherwise, treat it as a request for an experience.
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
    let latest_model = Arc::new(Mutex::new(None::<ModelUpdate>));

    let pull_buffer = Arc::clone(&replay_buffer);
    let rep_buffer = Arc::clone(&replay_buffer);
    let rep_latest_model = Arc::clone(&latest_model);
    let pub_latest_model = Arc::clone(&latest_model);
    let pull_latest_model = Arc::clone(&latest_model);

    let pull_task = pull_experiences(pull_socket, pull_buffer, pull_latest_model);
    let pub_task = publish_weights(pub_socket, pub_latest_model);
    let rep_task = run_rep_task(rep_socket, rep_buffer, rep_latest_model);

    join3(pull_task, pub_task, rep_task).await;
    Ok(())
}
