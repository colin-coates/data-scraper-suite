# MJ Data Scraper Suite

Enterprise-grade web scraping and data collection platform for Mountain Jewels Intelligence. Features anti-detection capabilities, plugin architecture, and comprehensive data validation.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│             SCRAPER ENGINE                      │
├─────────────────────────────────────────────────┤
│  🎯 Job Dispatcher (Async)                     │
│  🔌 Plugin Architecture                        │
│  🛡️  Anti-Detection Layer                      │
│  📊 Monitoring & Metrics                       │
├─────────────────────────────────────────────────┤
│  🕷️ SCRAPERS                                   │
│  • LinkedIn Scraper                            │
│  • Twitter/X Scraper                           │
│  • Company Website Scraper                     │
│  • News Article Scraper                        │
│  • Public Records Scraper                      │
├─────────────────────────────────────────────────┤
│  🎯 OUTPUT                                     │
│  • Azure Service Bus (Queue)                   │
│  • Azure Blob Storage (Raw Data)               │
│  • Structured JSON Format                      │
└─────────────────────────────────────────────────┘
```

## 🚀 Features

### Core Engine
- **Async Job Dispatcher**: Concurrent scraping with priority queuing
- **Plugin Architecture**: Easy addition of new scrapers
- **Anti-Detection Layer**: Dynamic headers, proxy rotation, human behavior simulation
- **Rate Limiting**: Intelligent throttling to avoid blocks
- **Error Recovery**: Automatic retry with exponential backoff

### Scrapers Included
- **LinkedIn Scraper**: Professional profile data collection
- **Twitter/X Scraper**: Social media data extraction
- **Company Website Scraper**: Corporate information gathering
- **News Article Scraper**: Media monitoring and sentiment analysis
- **Public Records Scraper**: Government and directory data

### API Collectors Included
- **LinkedIn API**: Official LinkedIn professional data access
- **Hunter.io API**: Email verification and contact discovery
- **Clearbit API**: Company enrichment and business intelligence
- **FullContact API**: Contact graph enrichment and social profiling

### Enterprise Features
- **Azure Integration**: Service Bus and Blob Storage connectivity
- **Comprehensive Logging**: Structured logging with correlation IDs
- **Health Monitoring**: Real-time performance metrics
- **Configuration Management**: YAML-based scraper configuration
- **Data Validation**: Pydantic models for data quality assurance

## 📦 Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## 🏃‍♂️ Quick Start

```python
from mj_data_scraper_suite.scraper_engine import ScraperEngine

# Initialize engine
engine = ScraperEngine()

# Add scraping job
await engine.dispatch_job({
    "scraper_type": "linkedin",
    "target": "john-doe-profile",
    "priority": "high"
})

# Start processing
await engine.start()
```

## 🔧 Configuration

Create `config/scrapers.yaml`:

```yaml
scrapers:
  linkedin:
    enabled: true
    rate_limit: 10  # requests per minute
    proxies:
      - http://proxy1:8080
      - http://proxy2:8080
    headers:
      user_agent_rotation: true
      cookie_persistence: true

  twitter:
    enabled: true
    rate_limit: 30
    api_keys:
      - key1
      - key2
```

## 🛡️ Anti-Detection Features

### Dynamic Headers
- Rotates User-Agent strings
- Randomizes Accept-Language headers
- Adds realistic Referer headers
- Includes custom headers for specific sites

### Human Behavior Simulation
- Random delays between requests (1-5 seconds)
- Mouse movement simulation
- Scroll behavior emulation
- Typing pattern simulation

### Cookie Persistence
- Maintains session cookies across requests
- Handles login sessions automatically
- Cookie jar management per domain

### Proxy Rotation
- Automatic proxy switching on failures
- Geographic proxy distribution
- Residential vs datacenter proxy balancing

## 📊 Monitoring

Access metrics at `/metrics`:

```json
{
  "scrapers_active": 5,
  "jobs_completed": 1247,
  "error_rate": 0.023,
  "avg_response_time": 2.3,
  "proxy_health": 0.98
}
```

## 🔌 Plugin Architecture

Add new scrapers easily:

```python
from mj_data_scraper_suite.core.base_scraper import BaseScraper

class CustomScraper(BaseScraper):
    async def scrape(self, target: dict) -> dict:
        # Your scraping logic here
        return {"data": "scraped_content"}

# Register plugin
engine.register_scraper("custom", CustomScraper)
```

## 📈 Performance

- **Concurrent Jobs**: Up to 50 simultaneous scrapers
- **Success Rate**: 95%+ with anti-detection enabled
- **Throughput**: 1000+ profiles per hour
- **Error Recovery**: 99.9% uptime with automatic retries

## 🔒 Security & Compliance

- **Rate Limiting**: Respects API limits and robots.txt
- **Data Encryption**: All data encrypted in transit and at rest
- **Audit Logging**: Complete activity tracking for compliance
- **GDPR Compliance**: Data minimization and consent management

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=mj_data_scraper_suite tests/

# Run specific scraper tests
pytest tests/test_linkedin_scraper.py
```

## 📚 API Documentation

Full API documentation available at `/docs` when running the service.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## 📄 License

Proprietary - Mountain Jewels Intelligence © 2024

## 🆘 Support

For support, contact engineering@mountainjewels.com