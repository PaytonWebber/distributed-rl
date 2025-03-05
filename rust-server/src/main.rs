use std::convert::TryInto;
use tokio;
use zeromq::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Start server");
    let mut socket = zeromq::RepSocket::new();
    socket.bind("tcp://0.0.0.0:5555").await?;

    loop {
        let mut repl: String = socket.recv().await?.try_into()?;
        println!("Receieved: {:?}", repl);
        repl.push_str(" Reply");
        socket.send(repl.into()).await?;
    }
}
