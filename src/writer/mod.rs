//! Writer layer - Database abstraction for blockchain ingestion
//!
//! Provides the GraphWriter trait and implementations for persisting blockchain
//! data to graph databases. This layer is isolated from database-specific
//! concerns, enabling testing without Neo4j and future database support.
//!
//! # Architecture
//! - `traits::GraphWriter` - Database abstraction trait
//! - `mock::MockWriter` - In-memory implementation for testing
//! - `neo4j::Neo4jWriter` - Production Neo4j implementation
//! - `csv::CsvWriter` - CSV file export implementation
//! - `error::WriterError` - Error types for writer operations
//!
//! # Example
//! ```
//! use bitcoin_chain_graph::writer::{GraphWriter, MockWriter};
//!
//! #[tokio::main]
//! async fn main() {
//!     let writer = MockWriter::new();
//!     writer.init_schema().await.unwrap();
//! }
//! ```

pub mod csv;
mod error;
pub mod mock;
pub mod neo4j;
mod traits;

pub use csv::CsvWriter;
pub use error::{Result, WriterError};
pub use mock::MockWriter;
pub use neo4j::Neo4jWriter;
pub use traits::GraphWriter;
