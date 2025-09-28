# 📊 Memorizer Framework - Monitoring & Observability

This document provides comprehensive information about the monitoring and observability features implemented in the Memorizer framework.

## 🎯 Overview

The Memorizer framework includes a complete monitoring and observability stack that provides:

- **Structured Logging** with request tracing and log aggregation
- **Performance Monitoring** with metrics collection and alerting
- **Health Checks** with dependency monitoring and automated testing
- **Real-time Dashboards** for system visualization
- **Distributed Tracing** for request correlation
- **Automated Testing** for system validation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                        │
├─────────────────────────────────────────────────────────────┤
│  Dashboard  │  Alerts  │  Metrics  │  Logs  │  Health     │
├─────────────────────────────────────────────────────────────┤
│  Tracing    │  Testing │  Cache    │  DB    │  External   │
├─────────────────────────────────────────────────────────────┤
│              Application Layer (FastAPI)                   │
├─────────────────────────────────────────────────────────────┤
│              Business Logic (Memory Manager)               │
├─────────────────────────────────────────────────────────────┤
│              Data Layer (Database, Cache, Vector DB)       │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Features

### 1. Comprehensive Logging

#### Structured Logging
- **JSON Format**: All logs are structured in JSON format for easy parsing
- **Request Tracing**: Every request gets a unique trace ID for correlation
- **Context Enrichment**: Logs include user ID, session ID, and operation context
- **Severity Levels**: CRITICAL, HIGH, MEDIUM, LOW with appropriate handling

#### Log Aggregation
- **Elasticsearch**: Full-text search and log analysis
- **Fluentd**: Log forwarding and processing
- **Splunk**: Enterprise log management
- **File Rotation**: Automatic log rotation with size limits

#### Request Tracing
- **Distributed Tracing**: Track requests across services
- **Performance Timing**: Measure request duration and bottlenecks
- **Error Correlation**: Link errors to specific requests and users

### 2. Performance Monitoring

#### Metrics Collection
- **Prometheus Integration**: Industry-standard metrics format
- **Custom Metrics**: Business and application-specific metrics
- **Real-time Collection**: Continuous metrics gathering
- **Historical Data**: Metrics retention and trending

#### Key Metrics
- **Request Metrics**: Count, duration, error rates
- **Memory Metrics**: Operations, tier distribution, performance
- **Database Metrics**: Query performance, connection pool stats
- **Cache Metrics**: Hit ratios, operation timing
- **Vector DB Metrics**: Embedding generation, query performance
- **System Metrics**: CPU, memory, disk usage

#### Alerting System
- **Configurable Rules**: Custom alert thresholds and conditions
- **Multiple Channels**: Email, Slack, Webhook notifications
- **Alert Management**: Acknowledgment and escalation
- **Severity Levels**: Critical, High, Medium, Low

### 3. Health Checks

#### Component Monitoring
- **Database Health**: Connection, query performance, pool status
- **Cache Health**: Connectivity, operation success rates
- **Vector DB Health**: Embedding generation, query performance
- **Memory Manager**: Core functionality validation
- **System Resources**: CPU, memory, disk usage
- **External Services**: API connectivity and response times

#### Health Check Types
- **Liveness Checks**: Is the service running?
- **Readiness Checks**: Is the service ready to handle requests?
- **Dependency Checks**: Are external dependencies available?
- **Performance Checks**: Are response times within acceptable limits?

#### Automated Testing
- **Component Tests**: Individual component validation
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmark and load testing
- **Regression Tests**: Ensure no functionality regression

### 4. Real-time Dashboard

#### Dashboard Features
- **System Overview**: Overall health and status
- **Performance Metrics**: Real-time charts and graphs
- **Alert Management**: Active alerts and acknowledgments
- **Log Viewer**: Recent log entries with filtering
- **Trace Explorer**: Request tracing and debugging

#### Visualization
- **Interactive Charts**: Request rates, response times, error rates
- **Status Indicators**: Component health with color coding
- **Trend Analysis**: Historical performance trends
- **Customizable Views**: Configurable dashboard layouts

## 🚀 Getting Started

### 1. Configuration

Copy the monitoring configuration template:
```bash
cp monitoring.env.example monitoring.env
```

Edit `monitoring.env` with your specific settings:
```bash
# Enable monitoring features
LOG_LEVEL=INFO
LOG_FORMAT=json
PERFORMANCE_MONITORING_ENABLED=true
HEALTH_CHECK_ENABLED=true
DASHBOARD_ENABLED=true
```

### 2. Start the Application

The monitoring system starts automatically with the application:
```bash
python -m src.api
```

### 3. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8000/dashboard/
```

### 4. View Metrics

Access Prometheus metrics:
```
http://localhost:8000/monitoring/prometheus
```

## 📊 Monitoring Endpoints

### Health Endpoints
- `GET /health` - Basic health check
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe
- `GET /monitoring/health/detailed` - Detailed health status

### Metrics Endpoints
- `GET /monitoring/performance` - Performance metrics summary
- `GET /monitoring/prometheus` - Prometheus-formatted metrics
- `GET /metrics` - Legacy metrics endpoint

### Testing Endpoints
- `GET /monitoring/tests` - Run all automated tests
- `GET /monitoring/tests/{component}` - Run component-specific tests

### Dashboard Endpoints
- `GET /dashboard/` - Main dashboard interface
- `GET /dashboard/api/data` - Dashboard data API
- `GET /dashboard/api/metrics` - Dashboard metrics API

## 🔧 Configuration Options

### Logging Configuration
```bash
# Log level and format
LOG_LEVEL=INFO
LOG_FORMAT=json

# Log files and rotation
LOG_FILE=logs/memorizer.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5

# Log aggregation
LOG_AGGREGATION_ENABLED=true
LOG_AGGREGATION_TYPE=elasticsearch
ELASTICSEARCH_HOSTS=localhost:9200
```

### Performance Monitoring
```bash
# Performance monitoring
PERFORMANCE_MONITORING_ENABLED=true
METRICS_COLLECTION_INTERVAL=30

# Performance thresholds
PERFORMANCE_THRESHOLD_REQUEST_TIME_MS=5000
PERFORMANCE_THRESHOLD_DATABASE_TIME_MS=1000
```

### Alerting Configuration
```bash
# Alerting
ALERTING_ENABLED=true
ALERT_ERROR_RATE_THRESHOLD=0.05
ALERT_RESPONSE_TIME_THRESHOLD=5.0

# Notification channels
ALERT_EMAIL_ENABLED=true
ALERT_SLACK_ENABLED=true
ALERT_WEBHOOK_ENABLED=true
```

### Health Checks
```bash
# Health check settings
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10

# Component-specific settings
HEALTH_CHECK_DATABASE_ENABLED=true
HEALTH_CHECK_CACHE_ENABLED=true
HEALTH_CHECK_VECTOR_DB_ENABLED=true
```

## 📈 Key Metrics

### Request Metrics
- `memorizer_requests_total` - Total request count by method, endpoint, status
- `memorizer_request_duration_seconds` - Request duration histogram
- `memorizer_request_errors_total` - Request error count

### Memory Metrics
- `memorizer_memory_operations_total` - Memory operations by type and tier
- `memorizer_memory_duration_seconds` - Memory operation duration
- `memorizer_memory_tier_count` - Memory count per tier

### Database Metrics
- `memorizer_database_operations_total` - Database operations by type
- `memorizer_database_duration_seconds` - Database operation duration
- `memorizer_active_connections` - Active database connections

### Cache Metrics
- `memorizer_cache_operations_total` - Cache operations by type
- `memorizer_cache_hit_ratio` - Cache hit ratio percentage

### System Metrics
- `memorizer_memory_usage_bytes` - Application memory usage
- `memorizer_cpu_usage_percent` - CPU usage percentage
- `memorizer_disk_usage_percent` - Disk usage percentage

## 🚨 Alert Rules

### Default Alert Rules
1. **High Error Rate**: Error rate > 5%
2. **High Response Time**: Response time > 5 seconds
3. **Low Cache Hit Ratio**: Cache hit ratio < 80%
4. **High Memory Usage**: Memory usage > 1GB
5. **Database Connection Issues**: Database health check failures
6. **External Service Failures**: External API connectivity issues

### Custom Alert Rules
You can add custom alert rules programmatically:
```python
from src.performance_monitor import get_performance_monitor, AlertRule

monitor = get_performance_monitor()
monitor.add_alert_rule(AlertRule(
    name="custom_alert",
    metric_name="custom_metric",
    condition=">",
    threshold=100.0,
    severity="high",
    description="Custom alert description"
))
```

## 🧪 Automated Testing

### Test Suites
1. **Database Tests**: Connection, operations, performance
2. **Cache Tests**: Connectivity, operations, hit ratios
3. **Memory Manager Tests**: Basic operations, tier management
4. **Vector DB Tests**: Embedding generation, query operations
5. **API Tests**: Health endpoints, memory endpoints
6. **Integration Tests**: End-to-end workflows
7. **Performance Tests**: Benchmark and load testing

### Running Tests
```bash
# Run all tests
curl http://localhost:8000/monitoring/tests

# Run component-specific tests
curl http://localhost:8000/monitoring/tests/database
curl http://localhost:8000/monitoring/tests/cache
curl http://localhost:8000/monitoring/tests/memory_manager
```

## 📊 Dashboard Usage

### Main Dashboard
The main dashboard provides:
- **System Status**: Overall health and component status
- **Key Metrics**: Request rate, error rate, response time
- **Performance Charts**: Real-time performance visualization
- **Alert Summary**: Active alerts and their status

### Custom Dashboards
You can create custom dashboards by:
1. Extending the dashboard service
2. Adding custom metrics
3. Creating custom visualizations
4. Configuring alert rules

## 🔍 Troubleshooting

### Common Issues

#### High Error Rates
1. Check application logs for error details
2. Review database connection pool status
3. Verify external service connectivity
4. Check system resource usage

#### Performance Issues
1. Monitor request duration metrics
2. Check database query performance
3. Review cache hit ratios
4. Analyze system resource usage

#### Health Check Failures
1. Review component-specific health checks
2. Check dependency connectivity
3. Verify configuration settings
4. Review automated test results

### Debug Mode
Enable debug mode for detailed logging:
```bash
LOG_LEVEL=DEBUG
DEBUG_MODE=true
VERBOSE_LOGGING=true
```

## 🔒 Security Considerations

### Log Security
- **Sensitive Data**: Avoid logging sensitive information
- **Access Control**: Restrict access to log files
- **Retention**: Implement appropriate log retention policies
- **Encryption**: Encrypt logs in transit and at rest

### Metrics Security
- **Access Control**: Restrict access to metrics endpoints
- **Data Privacy**: Avoid exposing sensitive business data
- **Rate Limiting**: Implement rate limiting for metrics endpoints

### Dashboard Security
- **Authentication**: Enable dashboard authentication
- **Authorization**: Implement role-based access control
- **HTTPS**: Use HTTPS for dashboard access
- **Network Security**: Restrict network access to dashboard

## 📚 Integration Examples

### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'memorizer'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/monitoring/prometheus'
    scrape_interval: 30s
```

### Grafana Dashboard
Import the provided Grafana dashboard configuration for comprehensive visualization.

### ELK Stack Integration
Configure Logstash to parse structured logs and send to Elasticsearch for analysis.

## 🚀 Production Deployment

### Monitoring Stack
For production deployment, consider:
1. **Centralized Logging**: ELK stack or similar
2. **Metrics Collection**: Prometheus with Grafana
3. **Alerting**: AlertManager with multiple notification channels
4. **Distributed Tracing**: Jaeger or Zipkin
5. **Health Monitoring**: Kubernetes health checks

### Scaling Considerations
- **Log Volume**: Plan for high log volumes
- **Metrics Storage**: Consider metrics retention policies
- **Alert Noise**: Tune alert rules to reduce false positives
- **Dashboard Performance**: Optimize dashboard queries

## 📞 Support

For monitoring-related issues:
1. Check the logs for error details
2. Review health check status
3. Run automated tests for validation
4. Consult the troubleshooting guide
5. Contact the development team

---

**Note**: This monitoring system is designed to provide comprehensive observability for the Memorizer framework. Regular monitoring and maintenance are essential for optimal performance and reliability.
