mod experience;
mod replay_buffer;
mod zmq_handler;

use futures::future::join3;
use replay_buffer::ReplayBuffer;
use std::sync::Arc;
use tokio;
use tokio::sync::Mutex;
use zeromq::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let replay_buffer = Arc::new(Mutex::new(ReplayBuffer::new(10000)));
    let model_params = Arc::new(Mutex::new(Vec::<u8>::new()));

    let mut pull_socket = PullSocket::new();
    pull_socket.bind("tcp://0.0.0.0:5555").await?;
    let pull_replay_buffer = Arc::clone(&replay_buffer);
    let pull_task = zmq_handler::pull_experiences(pull_socket, pull_replay_buffer);

    let mut rep_socket = RepSocket::new();
    rep_socket.bind("tcp://0.0.0.0:5556").await?;
    let rep_replay_buffer = Arc::clone(&replay_buffer);
    let rep_params_buffer = Arc::clone(&model_params);
    let rep_task = zmq_handler::rep_learner(rep_socket, rep_replay_buffer, rep_params_buffer);

    let mut pub_socket = PubSocket::new();
    pub_socket.bind("tcp://0.0.0.0:5557").await?;
    let pub_params_buffer = Arc::clone(&model_params);
    let pub_task = zmq_handler::publish_params(pub_socket, pub_params_buffer);

    join3(pull_task, rep_task, pub_task).await;

    Ok(())
}
