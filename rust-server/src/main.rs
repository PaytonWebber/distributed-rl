use tokio;
use zeromq::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting server ... ");
    let mut socket = zeromq::PullSocket::new();
    socket.bind("tcp://0.0.0.0:5555").await?;
    println!("Waiting for messages ... ");

    loop {
        let recv_message = socket.recv().await?;
        println!("Receieved: {:?}", recv_message);
    }
}
