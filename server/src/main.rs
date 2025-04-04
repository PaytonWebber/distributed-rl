mod experience;
mod replay_buffer;
mod zmq_handler;

use futures::future::join;
use replay_buffer::ReplayBuffer;
use std::sync::Arc;
use tokio;
use tokio::sync::Mutex;
use zeromq::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let replay_buffer = Arc::new(Mutex::new(ReplayBuffer::new(1000)));

    let mut pull_socket = PullSocket::new();
    pull_socket.bind("tcp://0.0.0.0:5555").await?;
    let pull_buffer = Arc::clone(&replay_buffer);
    let pull_task = zmq_handler::pull_experiences(pull_socket, pull_buffer);

    let mut rep_socket = RepSocket::new();
    rep_socket.bind("tcp://0.0.0.0:5556").await?;
    let rep_buffer = Arc::clone(&replay_buffer);
    let rep_task = zmq_handler::rep_task(rep_socket, rep_buffer);

    join(pull_task, rep_task).await;

    Ok(())
}
