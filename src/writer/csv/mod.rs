//! CSV implementation of GraphWriter trait
//!
//! Writes blockchain data to CSV files - one file per node label and one file per relationship type.
//! This enables easy data export and analysis without requiring a Neo4j database.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::Mutex;

use crate::domain::{
    BenefitsToData, BlockData, CheckpointData, InputData, OutputData, OutputLookupResult,
    PerformsData, TransactionData,
};
use crate::writer::{GraphWriter, Result, WriterError};

/// Default buffer size for file writers (512 KB). Larger buffers reduce syscalls.
const DEFAULT_BUFFER_CAPACITY: usize = 512 * 1024;

/// CSV writer that writes blockchain data to CSV files
///
/// Creates one CSV file per node label:
/// - Block.csv
/// - Transaction.csv
/// - Output.csv
/// - Input.csv
/// - Address.csv
/// - IngestionCheckpoint.csv
///
/// And one CSV file per relationship type:
/// - NEXT_BLOCK.csv
/// - INCLUDED_IN.csv
/// - HAS_OUTPUT.csv
/// - HAS_INPUT.csv
/// - SPENDS.csv
/// - LOCKED_TO.csv
/// - PERFORMS.csv
/// - BENEFITS_TO.csv
pub struct CsvWriter {
    /// Base directory for CSV files
    output_dir: PathBuf,
    /// Map of file paths to their initialized state (whether headers have been written)
    initialized_files: Arc<Mutex<HashMap<PathBuf, bool>>>,
    /// Map of file paths to their BufWriter handles
    writers: Arc<Mutex<HashMap<PathBuf, BufWriter<File>>>>,
}

impl CsvWriter {
    /// Create a new CsvWriter that writes to the specified directory
    ///
    /// # Arguments
    /// * `output_dir` - Directory path where CSV files will be created
    ///
    /// # Errors
    /// Returns error if the directory cannot be created or accessed
    pub async fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        tokio::fs::create_dir_all(&output_dir).await.map_err(|e| {
            WriterError::DatabaseError(format!("Failed to create output directory: {}", e))
        })?;

        Ok(Self {
            output_dir,
            initialized_files: Arc::new(Mutex::new(HashMap::new())),
            writers: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get or create a file writer for the given file path
    async fn get_writer(&self, file_path: &Path) -> Result<()> {
        let mut writers = self.writers.lock().await;

        if !writers.contains_key(file_path) {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)
                .await
                .map_err(|e| {
                    WriterError::DatabaseError(format!(
                        "Failed to open file {:?}: {}",
                        file_path, e
                    ))
                })?;

            writers.insert(
                file_path.to_path_buf(),
                BufWriter::with_capacity(DEFAULT_BUFFER_CAPACITY, file),
            );
        }

        Ok(())
    }

    /// Check if a file has been initialized (headers written)
    async fn is_initialized(&self, file_path: &Path) -> bool {
        let initialized = self.initialized_files.lock().await;
        initialized.get(file_path).copied().unwrap_or(false)
    }

    /// Mark a file as initialized (headers written)
    async fn mark_initialized(&self, file_path: &Path) {
        let mut initialized = self.initialized_files.lock().await;
        initialized.insert(file_path.to_path_buf(), true);
    }

    /// Write CSV headers if the file hasn't been initialized yet
    async fn write_headers_if_needed(&self, file_path: &Path, headers: &[&str]) -> Result<()> {
        if !self.is_initialized(file_path).await {
            self.get_writer(file_path).await?;

            let mut writers = self.writers.lock().await;
            if let Some(writer) = writers.get_mut(file_path) {
                let header_line = headers.join(",") + "\n";
                writer
                    .write_all(header_line.as_bytes())
                    .await
                    .map_err(|e| {
                        WriterError::DatabaseError(format!(
                            "Failed to write headers to {:?}: {}",
                            file_path, e
                        ))
                    })?;

                writer.flush().await.map_err(|e| {
                    WriterError::DatabaseError(format!(
                        "Failed to flush headers to {:?}: {}",
                        file_path, e
                    ))
                })?;
            }

            self.mark_initialized(file_path).await;
        }

        Ok(())
    }

    /// Write many CSV rows in a single lock and syscall. Much faster than per-row writes.
    async fn write_rows_batch(&self, file_path: &Path, rows: &[Vec<String>]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        self.get_writer(file_path).await?;

        let body: String = rows
            .iter()
            .map(|row| row.join(","))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";

        let mut writers = self.writers.lock().await;
        if let Some(writer) = writers.get_mut(file_path) {
            writer.write_all(body.as_bytes()).await.map_err(|e| {
                WriterError::DatabaseError(format!(
                    "Failed to write batch to {:?}: {}",
                    file_path, e
                ))
            })?;
        }

        Ok(())
    }

    /// Flush all writers (useful for checkpointing)
    async fn flush_all(&self) -> Result<()> {
        let mut writers = self.writers.lock().await;
        for writer in writers.values_mut() {
            writer.flush().await.map_err(|e| {
                WriterError::DatabaseError(format!("Failed to flush writer: {}", e))
            })?;
        }
        Ok(())
    }

    /// Escape a CSV field value (handles commas, quotes, newlines)
    fn escape_csv_field(value: &str) -> String {
        if value.contains(',') || value.contains('"') || value.contains('\n') {
            format!("\"{}\"", value.replace("\"", "\"\""))
        } else {
            value.to_string()
        }
    }

    /// Format an optional value for CSV
    fn format_optional<T: std::fmt::Display>(value: &Option<T>) -> String {
        match value {
            Some(v) => v.to_string(),
            None => String::new(),
        }
    }
}

#[async_trait]
impl GraphWriter for CsvWriter {
    async fn init_schema(&self) -> Result<()> {
        // CSV files don't need schema initialization - headers are written on first write
        Ok(())
    }

    async fn write_blocks(&self, blocks: &[BlockData]) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        let file_path = self.output_dir.join("Block.csv");

        // Write headers if needed
        self.write_headers_if_needed(
            &file_path,
            &[
                "height",
                "hash",
                "previousHash",
                "merkleRoot",
                "timestamp",
                "bits",
                "difficulty",
                "nonce",
                "version",
                "txCount",
                "size",
                "weight",
            ],
        )
        .await?;

        // Write blocks (batched)
        let block_rows: Vec<Vec<String>> = blocks
            .iter()
            .map(|block| {
                vec![
                    block.height.to_string(),
                    Self::escape_csv_field(&block.hash),
                    Self::escape_csv_field(&block.previous_hash),
                    Self::escape_csv_field(&block.merkle_root),
                    block.timestamp.to_string(),
                    Self::escape_csv_field(&block.bits),
                    block.difficulty.to_string(),
                    block.nonce.to_string(),
                    block.version.to_string(),
                    block.tx_count.to_string(),
                    block.size.to_string(),
                    block.weight.to_string(),
                ]
            })
            .collect();
        self.write_rows_batch(&file_path, &block_rows).await?;

        // Write NEXT_BLOCK relationships (batched)
        let rel_file_path = self.output_dir.join("NEXT_BLOCK.csv");
        self.write_headers_if_needed(&rel_file_path, &["from_hash", "to_hash"])
            .await?;

        let next_block_rows: Vec<Vec<String>> = blocks
            .iter()
            .filter(|b| b.height > 0)
            .map(|block| {
                vec![
                    Self::escape_csv_field(&block.previous_hash),
                    Self::escape_csv_field(&block.hash),
                ]
            })
            .collect();
        self.write_rows_batch(&rel_file_path, &next_block_rows).await?;

        Ok(())
    }

    async fn write_transactions(&self, transactions: &[TransactionData]) -> Result<()> {
        if transactions.is_empty() {
            return Ok(());
        }

        let file_path = self.output_dir.join("Transaction.csv");

        // Write headers if needed
        self.write_headers_if_needed(
            &file_path,
            &[
                "txid",
                "blockHeight",
                "blockHash",
                "timestamp",
                "version",
                "locktime",
                "size",
                "vsize",
                "weight",
                "isCoinbase",
                "totalInput",
                "totalOutput",
                "fee",
            ],
        )
        .await?;

        // Write transactions (batched)
        let tx_rows: Vec<Vec<String>> = transactions
            .iter()
            .map(|tx| {
                vec![
                    Self::escape_csv_field(&tx.txid),
                    tx.block_height.to_string(),
                    Self::escape_csv_field(&tx.block_hash),
                    tx.timestamp.to_string(),
                    tx.version.to_string(),
                    tx.locktime.to_string(),
                    tx.size.to_string(),
                    tx.vsize.to_string(),
                    tx.weight.to_string(),
                    tx.is_coinbase.to_string(),
                    Self::format_optional(&tx.total_input),
                    Self::format_optional(&tx.total_output),
                    Self::format_optional(&tx.fee),
                ]
            })
            .collect();
        self.write_rows_batch(&file_path, &tx_rows).await?;

        // Write INCLUDED_IN relationships (batched)
        let rel_file_path = self.output_dir.join("INCLUDED_IN.csv");
        self.write_headers_if_needed(&rel_file_path, &["from_txid", "to_block_hash"])
            .await?;

        let included_in_rows: Vec<Vec<String>> = transactions
            .iter()
            .map(|tx| {
                vec![
                    Self::escape_csv_field(&tx.txid),
                    Self::escape_csv_field(&tx.block_hash),
                ]
            })
            .collect();
        self.write_rows_batch(&rel_file_path, &included_in_rows).await?;

        Ok(())
    }

    async fn write_outputs(&self, outputs: &[OutputData]) -> Result<()> {
        if outputs.is_empty() {
            return Ok(());
        }

        let file_path = self.output_dir.join("Output.csv");

        // Write headers if needed
        self.write_headers_if_needed(
            &file_path,
            &[
                "outputId",
                "outputIndex",
                "txid",
                "amount",
                "scriptPubKey",
                "scriptType",
                "address",
            ],
        )
        .await?;

        // Write outputs (batched)
        let output_rows: Vec<Vec<String>> = outputs
            .iter()
            .map(|output| {
                vec![
                    Self::escape_csv_field(&output.output_id),
                    output.output_index.to_string(),
                    Self::escape_csv_field(&output.txid),
                    output.amount.to_string(),
                    Self::escape_csv_field(&output.script_pubkey),
                    Self::escape_csv_field(&output.script_type),
                    Self::format_optional(&output.address),
                ]
            })
            .collect();
        self.write_rows_batch(&file_path, &output_rows).await?;

        // Write LOCKED_TO relationships and Address nodes (batched)
        let rel_file_path = self.output_dir.join("LOCKED_TO.csv");
        self.write_headers_if_needed(&rel_file_path, &["from_outputId", "to_address"])
            .await?;

        let addr_file_path = self.output_dir.join("Address.csv");
        self.write_headers_if_needed(&addr_file_path, &["address"])
            .await?;

        let mut seen_addresses = std::collections::HashSet::new();
        let mut addr_rows = Vec::new();
        let mut locked_to_rows = Vec::new();

        for output in outputs {
            if let Some(ref address) = output.address {
                if seen_addresses.insert(address.clone()) {
                    addr_rows.push(vec![Self::escape_csv_field(address)]);
                }
                locked_to_rows.push(vec![
                    Self::escape_csv_field(&output.output_id),
                    Self::escape_csv_field(address),
                ]);
            }
        }

        if !addr_rows.is_empty() {
            self.write_rows_batch(&addr_file_path, &addr_rows).await?;
        }
        if !locked_to_rows.is_empty() {
            self.write_rows_batch(&rel_file_path, &locked_to_rows).await?;
        }

        Ok(())
    }

    async fn write_has_output_relationships(&self, outputs: &[OutputData]) -> Result<()> {
        if outputs.is_empty() {
            return Ok(());
        }

        let rel_file_path = self.output_dir.join("HAS_OUTPUT.csv");
        self.write_headers_if_needed(&rel_file_path, &["from_txid", "to_outputId"])
            .await?;

        let has_output_rows: Vec<Vec<String>> = outputs
            .iter()
            .map(|output| {
                vec![
                    Self::escape_csv_field(&output.txid),
                    Self::escape_csv_field(&output.output_id),
                ]
            })
            .collect();
        self.write_rows_batch(&rel_file_path, &has_output_rows).await?;

        Ok(())
    }

    async fn write_inputs(&self, inputs: &[InputData]) -> Result<()> {
        if inputs.is_empty() {
            return Ok(());
        }

        let file_path = self.output_dir.join("Input.csv");

        // Write headers if needed
        self.write_headers_if_needed(
            &file_path,
            &[
                "inputId",
                "inputIndex",
                "txid",
                "previousTxid",
                "previousOutputIndex",
                "scriptSig",
                "sequence",
                "witness",
                "blockHeight",
            ],
        )
        .await?;

        // Write inputs (batched)
        let input_rows: Vec<Vec<String>> = inputs
            .iter()
            .map(|input| {
                let witness_str = if input.witness.is_empty() {
                    String::new()
                } else {
                    input.witness.join(";")
                };
                vec![
                    Self::escape_csv_field(&input.input_id),
                    input.input_index.to_string(),
                    Self::escape_csv_field(&input.txid),
                    Self::escape_csv_field(&input.previous_txid),
                    input.previous_output_index.to_string(),
                    Self::escape_csv_field(&input.script_sig),
                    input.sequence.to_string(),
                    Self::escape_csv_field(&witness_str),
                    input.block_height.to_string(),
                ]
            })
            .collect();
        self.write_rows_batch(&file_path, &input_rows).await?;

        // Write HAS_INPUT relationships (batched)
        let has_input_file_path = self.output_dir.join("HAS_INPUT.csv");
        self.write_headers_if_needed(&has_input_file_path, &["from_txid", "to_inputId"])
            .await?;

        let has_input_rows: Vec<Vec<String>> = inputs
            .iter()
            .map(|input| {
                vec![
                    Self::escape_csv_field(&input.txid),
                    Self::escape_csv_field(&input.input_id),
                ]
            })
            .collect();
        self.write_rows_batch(&has_input_file_path, &has_input_rows).await?;

        // Write SPENDS relationships (batched, skip coinbase)
        let spends_file_path = self.output_dir.join("SPENDS.csv");
        self.write_headers_if_needed(&spends_file_path, &["from_inputId", "to_outputId"])
            .await?;

        let spends_rows: Vec<Vec<String>> = inputs
            .iter()
            .filter(|i| i.previous_output_index != 0xFFFFFFFF)
            .map(|input| {
                let previous_output_id =
                    format!("{}:{}", input.previous_txid, input.previous_output_index);
                vec![
                    Self::escape_csv_field(&input.input_id),
                    Self::escape_csv_field(&previous_output_id),
                ]
            })
            .collect();
        self.write_rows_batch(&spends_file_path, &spends_rows).await?;

        Ok(())
    }

    async fn write_performs(&self, performs: &[PerformsData]) -> Result<()> {
        if performs.is_empty() {
            return Ok(());
        }

        let rel_file_path = self.output_dir.join("PERFORMS.csv");
        self.write_headers_if_needed(
            &rel_file_path,
            &["from_address", "to_txid", "inputCount", "amountSpent"],
        )
        .await?;

        // Ensure Address nodes exist (they should already exist from LOCKED_TO, but create if not)
        let addr_file_path = self.output_dir.join("Address.csv");
        self.write_headers_if_needed(&addr_file_path, &["address"])
            .await?;

        let mut seen_addresses = std::collections::HashSet::new();
        let mut addr_rows = Vec::new();
        let mut performs_rows = Vec::new();

        for perform in performs {
            if seen_addresses.insert(perform.from_address.clone()) {
                addr_rows.push(vec![Self::escape_csv_field(&perform.from_address)]);
            }
            performs_rows.push(vec![
                Self::escape_csv_field(&perform.from_address),
                Self::escape_csv_field(&perform.to_txid),
                perform.input_count.to_string(),
                perform.amount_spent.to_string(),
            ]);
        }

        if !addr_rows.is_empty() {
            self.write_rows_batch(&addr_file_path, &addr_rows).await?;
        }
        self.write_rows_batch(&rel_file_path, &performs_rows).await?;

        Ok(())
    }

    async fn write_benefits_to(&self, benefits_to: &[BenefitsToData]) -> Result<()> {
        if benefits_to.is_empty() {
            return Ok(());
        }

        let rel_file_path = self.output_dir.join("BENEFITS_TO.csv");
        self.write_headers_if_needed(
            &rel_file_path,
            &["from_txid", "to_address", "outputCount", "amountReceived"],
        )
        .await?;

        // Ensure Address nodes exist
        let addr_file_path = self.output_dir.join("Address.csv");
        self.write_headers_if_needed(&addr_file_path, &["address"])
            .await?;

        let mut seen_addresses = std::collections::HashSet::new();
        let mut addr_rows = Vec::new();
        let mut benefits_rows = Vec::new();

        for benefit in benefits_to {
            if seen_addresses.insert(benefit.to_address.clone()) {
                addr_rows.push(vec![Self::escape_csv_field(&benefit.to_address)]);
            }
            benefits_rows.push(vec![
                Self::escape_csv_field(&benefit.from_txid),
                Self::escape_csv_field(&benefit.to_address),
                benefit.output_count.to_string(),
                benefit.amount_received.to_string(),
            ]);
        }

        if !addr_rows.is_empty() {
            self.write_rows_batch(&addr_file_path, &addr_rows).await?;
        }
        self.write_rows_batch(&rel_file_path, &benefits_rows).await?;

        Ok(())
    }

    async fn mark_output_spent(
        &self,
        _output_id: &str,
        _spent_in_txid: &str,
        _spent_at_height: u32,
    ) -> Result<()> {
        // CSV files don't support updates - we'd need to read, modify, and rewrite
        // For now, we'll just return Ok() as this is mainly used for UTXO cache fallback
        // In a production CSV export, you might want to track this differently
        Ok(())
    }

    async fn create_checkpoint(&self) -> Result<()> {
        let checkpoint = CheckpointData {
            last_processed_height: -999,
            last_processed_hash: String::from(
                "0000000000000000000000000000000000000000000000000000000000000000",
            ),
            last_processed_file: String::from("blk00000.dat"),
            last_processed_file_offset: Some(0),
            timestamp: chrono::Utc::now().timestamp(),
            status: String::from("in_progress"),
        };
        self.update_checkpoint(&checkpoint).await
    }

    async fn update_checkpoint(&self, checkpoint: &CheckpointData) -> Result<()> {
        let file_path = self.output_dir.join("IngestionCheckpoint.csv");

        // For checkpoint, we'll overwrite the file each time (only one checkpoint)
        // First, clear existing content by truncating
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await
            .map_err(|e| {
                WriterError::DatabaseError(format!("Failed to open checkpoint file: {}", e))
            })?;

        let mut writer = BufWriter::new(file);

        // Write headers
        writer
            .write_all(b"lastProcessedHeight,lastProcessedHash,lastProcessedFile,lastProcessedFileOffset,timestamp,status\n")
            .await
            .map_err(|e| WriterError::DatabaseError(format!("Failed to write checkpoint headers: {}", e)))?;

        // Write checkpoint data
        let row = vec![
            checkpoint.last_processed_height.to_string(),
            Self::escape_csv_field(&checkpoint.last_processed_hash),
            Self::escape_csv_field(&checkpoint.last_processed_file),
            Self::format_optional(&checkpoint.last_processed_file_offset),
            checkpoint.timestamp.to_string(),
            Self::escape_csv_field(&checkpoint.status),
        ];
        let row_line = row.join(",") + "\n";
        writer.write_all(row_line.as_bytes()).await.map_err(|e| {
            WriterError::DatabaseError(format!("Failed to write checkpoint: {}", e))
        })?;

        writer.flush().await.map_err(|e| {
            WriterError::DatabaseError(format!("Failed to flush checkpoint: {}", e))
        })?;

        Ok(())
    }

    async fn get_checkpoint(&self) -> Result<Option<CheckpointData>> {
        let file_path = self.output_dir.join("IngestionCheckpoint.csv");

        if !file_path.exists() {
            return Ok(None);
        }

        // Read checkpoint file
        let content = tokio::fs::read_to_string(&file_path).await.map_err(|e| {
            WriterError::DatabaseError(format!("Failed to read checkpoint file: {}", e))
        })?;

        let mut lines = content.lines();

        // Skip header
        if lines.next().is_none() {
            return Ok(None);
        }

        // Read data line
        if let Some(data_line) = lines.next() {
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(data_line.as_bytes());

            if let Some(result) = reader.records().next() {
                let record = result.map_err(|e| {
                    WriterError::DatabaseError(format!("Failed to parse checkpoint CSV: {}", e))
                })?;

                if record.len() >= 6 {
                    let checkpoint = CheckpointData {
                        last_processed_height: record[0].parse().unwrap_or(-999),
                        last_processed_hash: record[1].to_string(),
                        last_processed_file: record[2].to_string(),
                        last_processed_file_offset: if record[3].is_empty() {
                            None
                        } else {
                            record[3].parse().ok()
                        },
                        timestamp: record[4].parse().unwrap_or(0),
                        status: record[5].to_string(),
                    };
                    return Ok(Some(checkpoint));
                }
            }
        }

        Ok(None)
    }

    async fn mark_checkpoint_complete(&self) -> Result<()> {
        if let Some(mut checkpoint) = self.get_checkpoint().await? {
            checkpoint.status = String::from("completed");
            checkpoint.timestamp = chrono::Utc::now().timestamp();
            self.update_checkpoint(&checkpoint).await
        } else {
            Err(WriterError::CheckpointError(
                "No checkpoint exists to mark as complete".to_string(),
            ))
        }
    }

    async fn set_checkpoint_status(&self, status: &str) -> Result<()> {
        if let Some(mut checkpoint) = self.get_checkpoint().await? {
            checkpoint.status = status.to_string();
            checkpoint.timestamp = chrono::Utc::now().timestamp();
            self.update_checkpoint(&checkpoint).await
        } else {
            Err(WriterError::CheckpointError(
                "No checkpoint exists to update status".to_string(),
            ))
        }
    }

    async fn lookup_outputs_batch(&self, output_ids: &[String]) -> Result<Vec<OutputLookupResult>> {
        let file_path = self.output_dir.join("Output.csv");

        if !file_path.exists() {
            return Ok(Vec::new());
        }

        // Read all outputs from CSV
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .map_err(|e| WriterError::DatabaseError(format!("Failed to read Output.csv: {}", e)))?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(content.as_bytes());

        let mut results = Vec::new();
        let requested_set: std::collections::HashSet<String> = output_ids.iter().cloned().collect();

        for result in reader.records() {
            let record = result.map_err(|e| {
                WriterError::DatabaseError(format!("Failed to parse Output.csv: {}", e))
            })?;

            if record.len() >= 7 {
                let output_id = record[0].to_string();
                if requested_set.contains(&output_id) {
                    let output = OutputLookupResult {
                        output_id: output_id.clone(),
                        output_index: record[1].parse().unwrap_or(0),
                        amount: record[3].parse().unwrap_or(0),
                        script_type: record[5].to_string(),
                        address: if record[6].is_empty() {
                            None
                        } else {
                            Some(record[6].to_string())
                        },
                    };
                    results.push(output);
                }
            }
        }

        Ok(results)
    }

    async fn lookup_block_hash(&self, height: u32) -> Result<Option<String>> {
        let file_path = self.output_dir.join("Block.csv");

        if !file_path.exists() {
            return Ok(None);
        }

        // Read blocks from CSV
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .map_err(|e| WriterError::DatabaseError(format!("Failed to read Block.csv: {}", e)))?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(content.as_bytes());

        for result in reader.records() {
            let record = result.map_err(|e| {
                WriterError::DatabaseError(format!("Failed to parse Block.csv: {}", e))
            })?;

            if record.len() >= 2 {
                if let Ok(block_height) = record[0].parse::<u32>() {
                    if block_height == height {
                        return Ok(Some(record[1].to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn rollback_block(&self, _height: u32) -> Result<()> {
        // CSV files don't support efficient deletion - would require rewriting entire files
        // For CSV export, rollback is not practical. Return error to indicate this limitation.
        Err(WriterError::DatabaseError(
            "Rollback not supported for CSV export. CSV files are append-only.".to_string(),
        ))
    }

    async fn get_max_block_height(&self) -> Result<Option<u32>> {
        let file_path = self.output_dir.join("Block.csv");

        if !file_path.exists() {
            return Ok(None);
        }

        // Read blocks from CSV and find max height
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .map_err(|e| WriterError::DatabaseError(format!("Failed to read Block.csv: {}", e)))?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(content.as_bytes());

        let mut max_height: Option<u32> = None;

        for result in reader.records() {
            let record = result.map_err(|e| {
                WriterError::DatabaseError(format!("Failed to parse Block.csv: {}", e))
            })?;

            if record.len() >= 1 {
                if let Ok(height) = record[0].parse::<u32>() {
                    max_height = Some(max_height.map(|h| h.max(height)).unwrap_or(height));
                }
            }
        }

        Ok(max_height)
    }

    async fn check_block_complete(&self, height: u32) -> Result<(u32, u32)> {
        let block_file_path = self.output_dir.join("Block.csv");
        let tx_file_path = self.output_dir.join("Transaction.csv");

        if !block_file_path.exists() {
            return Err(WriterError::QueryFailed(format!(
                "Block {} not found",
                height
            )));
        }

        // Find block and get expected tx count
        let block_content = tokio::fs::read_to_string(&block_file_path)
            .await
            .map_err(|e| WriterError::DatabaseError(format!("Failed to read Block.csv: {}", e)))?;

        let mut block_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(block_content.as_bytes());

        let mut expected_tx_count: Option<u32> = None;

        for result in block_reader.records() {
            let record = result.map_err(|e| {
                WriterError::DatabaseError(format!("Failed to parse Block.csv: {}", e))
            })?;

            if record.len() >= 10 {
                if let Ok(block_height) = record[0].parse::<u32>() {
                    if block_height == height {
                        expected_tx_count = record[9].parse().ok();
                        break;
                    }
                }
            }
        }

        let expected = expected_tx_count
            .ok_or_else(|| WriterError::QueryFailed(format!("Block {} not found", height)))?;

        // Count transactions in this block
        if !tx_file_path.exists() {
            return Ok((expected, 0));
        }

        let tx_content = tokio::fs::read_to_string(&tx_file_path)
            .await
            .map_err(|e| {
                WriterError::DatabaseError(format!("Failed to read Transaction.csv: {}", e))
            })?;

        let mut tx_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(tx_content.as_bytes());

        let mut actual_count = 0;

        for result in tx_reader.records() {
            let record = result.map_err(|e| {
                WriterError::DatabaseError(format!("Failed to parse Transaction.csv: {}", e))
            })?;

            if record.len() >= 2 {
                if let Ok(tx_block_height) = record[1].parse::<u32>() {
                    if tx_block_height == height {
                        actual_count += 1;
                    }
                }
            }
        }

        Ok((expected, actual_count))
    }

    async fn begin_transaction(&self) -> Result<()> {
        // CSV files don't support transactions - just flush to ensure data is written
        self.flush_all().await
    }

    async fn commit_transaction(&self) -> Result<()> {
        // CSV files don't support transactions - just flush to ensure data is written
        self.flush_all().await
    }

    async fn rollback_transaction(&self) -> Result<()> {
        // CSV files don't support transactions - return error
        Err(WriterError::TransactionFailed(
            "Rollback not supported for CSV export. CSV files are append-only.".to_string(),
        ))
    }
}
