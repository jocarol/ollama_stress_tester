# Ollama Stress Tester

This script is designed to stress test an Ollama model by simulating multiple customers and delivery personnel interacting with it concurrently. It measures response times, monitors system resource usage (CPU and memory), and provides real-time feedback on the test's progress. The tool is useful for evaluating the performance and scalability of an Ollama model under load.

![Live Dashboard](https://i.imgur.com/Vo6sz6R.png)


## Features

- Simulates multiple customers placing orders and checking their status.
- Simulates delivery personnel starting shifts, getting orders, and delivering them.
- Monitors CPU and memory usage during the test.
- Displays real-time metrics, active requests, and logs using the [Rich](https://github.com/Textualize/rich) library.
- Logs detailed information for debugging and analysis.

## Prerequisites

- Python 3.x
- An instance of Ollama running on `http://localhost:11434` (or another URL, configurable)
- Required Python libraries: `requests`, `psutil`, `rich`

## Installation

1. Clone or download this repository.
2. Install the required Python libraries:

   ```bash
   pip install requests psutil rich
   ```

3. Ensure that Ollama is running and accessible at the specified URL (default: `http://localhost:11434`).

## Configuration

The script has several configurable parameters at the top of the file:

- `OLLAMA_URL`: The URL of the Ollama API (default: `"http://localhost:11434/api/generate"`).
- `MODEL_NAME`: The name of the model to test (default: `"chikaka-customer-hermes2:latest"`).
- `NUM_CUSTOMERS`: Number of simulated customers (default: `5`).
- `NUM_DELIVERY`: Number of simulated delivery personnel (default: `2`).
- `TEST_DURATION`: Duration of the test in seconds (default: `600`).
- `INTERVAL_CUSTOMER`: Interval between customer actions in seconds (default: `10`).
- `INTERVAL_DELIVERY`: Interval between delivery personnel actions in seconds (default: `2`).

Adjust these parameters as needed for your testing scenario.

## Usage

To run the stress test, execute the script with Python:

```bash
python stress_test.py
```

The script will start simulating customer and delivery personnel actions, monitor system resources, and display a live dashboard with metrics, active requests, and logs. Below is an example of the live dashboard:

### Dashboard Sections

- **Header**: Displays the test configuration, including the model name, Ollama URL, number of customers and delivery personnel, test duration, and intervals.
- **Metrics**: Shows real-time statistics such as the number of customer and delivery actions, average response times, active and queued requests, CPU and memory usage, and tokens per second.
- **Active Requests**: Lists currently active requests, including the user type (Customer or Delivery), the action being performed, and the partial response received so far.
- **Logs**: Displays a log of recent actions and events, such as orders placed, status checks, and deliveries.
- **Progress**: A progress bar indicating the test's progress over the specified duration.

### Output

After the test completes, a summary report is printed to the console, including:

- Number of simulated customers and delivery personnel.
- Test duration.
- Average and maximum response times for customers and delivery personnel.
- Average and maximum CPU and memory usage.

Additionally, detailed logs are saved to `stress_debug.log` for further analysis.

## Interpreting the Results

- **Response Times**: Lower response times indicate better performance. High response times may suggest the model is struggling under load.
- **CPU and Memory Usage**: Monitor these to ensure the system isn't being overloaded. High usage might indicate the need for more resources or optimization.
- **Tokens Per Second**: This metric shows the rate at which the model is generating tokens. A higher rate indicates better throughput.

## Troubleshooting

- If the script times out or fails to connect to Ollama, ensure Ollama is running and the URL is correct.
- Check the `stress_debug.log` file for detailed logs and error messages.
- Adjust the number of simulated users or intervals if the system becomes overloaded.

## Customization

You can modify the prompts and actions in the `Customer` and `DeliveryPersonnel` classes to simulate different scenarios or test specific features of the model. For example, you can change the prompts to test different types of queries or adjust the frequency of actions.

## Conclusion

This stress testing tool provides a comprehensive way to evaluate the performance and scalability of an Ollama model under concurrent user loads. By adjusting the configuration and monitoring the metrics, you can identify bottlenecks and optimize your setup for better performance.
