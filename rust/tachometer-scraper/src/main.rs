fn main() -> Result<(), Box<dyn std::error::Error>> {
    tachometer_scraper::run_cli(std::env::args().skip(1))
}
