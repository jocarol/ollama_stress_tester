import threading
import time
import requests
import psutil
import statistics
import re
import json
from datetime import datetime
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich.layout import Layout
from rich.progress import Progress
from rich import box

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "chikaka-customer-hermes2:latest"
NUM_CUSTOMERS = 5
NUM_DELIVERY = 2
TEST_DURATION = 600
INTERVAL_CUSTOMER = 10
INTERVAL_DELIVERY = 2

# Data structures
lock = threading.Lock()
token_lock = threading.Lock()
customer_response_times = []
delivery_response_times = []
cpu_usages = []
memory_usages = []
active_requests = {}
logs = []
total_tokens = 0
tokens_per_second = 0

# Open log file
log_file = open("stress_debug.log", "w")

# Initialize console
console = Console()

def send_prompt(user_id, prompt):
    """Send a streaming prompt to the model, update active_requests, and return response and time."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "num_predict": 200,
        "stop": ["FIN_DE_REPONSE"]
    }
    start_time = time.time()
    accumulated_text = ""
    chunk_count = 0
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600) as response:
            for line in response.iter_lines():
                if line:
                    chunk_count += 1
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        chunk_text = chunk.get("response", "")
                        accumulated_text += chunk_text
                        num_tokens = len(chunk_text.split())
                        with token_lock:
                            global total_tokens
                            total_tokens += num_tokens
                        with lock:
                            if user_id in active_requests:
                                active_requests[user_id][2] = accumulated_text
                    except json.JSONDecodeError:
                        log_file.write(f"Invalid JSON in response line: {line}\n")
                        log_file.flush()
        duration = time.time() - start_time
        log_file.write(f"Request for user {user_id} completed in {duration:.2f} seconds with {chunk_count} chunks. Response: {accumulated_text}\n")
        log_file.flush()
        return accumulated_text, duration
    except requests.exceptions.Timeout:
        log_file.write(f"Request for user {user_id} timed out after 600 seconds\n")
        log_file.flush()
        return "Timeout", 600
    except Exception as e:
        log_file.write(f"Error during request for user {user_id}: {e}\n")
        log_file.flush()
        return "Error", 0

def perform_action(user_id, user_type, action, prompt, parse_response=None):
    """Helper function to perform an action, send prompt, and log results."""
    with lock:
        active_requests[user_id] = [user_type, action, ""]
    try:
        response, rt = send_prompt(user_id, prompt)
        response = response.strip()
        if response in ["Timeout", "Error"]:
            with lock:
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"{user_type} {user_id}",
                    "action": f"{action} failed: {response}"
                })
            return rt, None
        parsed = parse_response(response) if parse_response else None
        if parse_response:
            if parsed:
                with lock:
                    logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "user_id": f"{user_type} {user_id}",
                        "action": f"{action}: {parsed}"
                    })
            else:
                with lock:
                    logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "user_id": f"{user_type} {user_id}",
                        "action": f"{action} failed: Response unexpected. Got: '{response}'"
                    })
        else:
            with lock:
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"{user_type} {user_id}",
                    "action": action
                })
        return rt, parsed
    finally:
        with lock:
            if user_id in active_requests:
                del active_requests[user_id]

class Customer:
    def __init__(self, user_id):
        self.user_id = user_id
        self.order_id = None

    def place_order(self):
        def parse_order_response(response):
            match = re.search(r"Votre numéro de commande est (\d+)", response)
            return match.group(1) if match else None
        rt, order_id = perform_action(self.user_id, "Customer", "Placed order", 
                                      "En tant que client, je veux passer une commande pour une pizza.", 
                                      parse_order_response)
        if order_id:
            self.order_id = order_id
        with lock:
            customer_response_times.append(rt)
        return rt

    def check_status(self):
        if not self.order_id:
            return 0
        prompt = f"En tant que client, où en est ma commande {self.order_id} ?"
        rt, _ = perform_action(self.user_id, "Customer", f"Checked status of order {self.order_id}", prompt)
        with lock:
            customer_response_times.append(rt)
        return rt

class DeliveryPersonnel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.current_order_id = None

    def start_shift(self):
        rt, _ = perform_action(self.user_id, "Delivery", "Started shift", 
                               "En tant que livreur, je veux commencer mon quart.")
        with lock:
            delivery_response_times.append(rt)
        return rt

    def get_next_order(self):
        def parse_order_response(response):
            match = re.search(r"Vous avez une nouvelle commande\s*:\s*(\d+)", response, re.IGNORECASE)
            return match.group(1) if match else None
        rt, order_id = perform_action(self.user_id, "Delivery", "Got next order", 
                                      "En tant que livreur, je suis prêt pour la prochaine commande.", 
                                      parse_order_response)
        if order_id:
            self.current_order_id = order_id
        elif "Aucune commande disponible pour le moment." in rt[1] or "":
            with lock:
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"Delivery {self.user_id}",
                    "action": "No orders available at the moment"
                })
        with lock:
            delivery_response_times.append(rt)
        return rt

    def deliver_order(self):
        if not self.current_order_id:
            return 0
        prompt = f"En tant que livreur, j'ai livré la commande {self.current_order_id}."
        rt, _ = perform_action(self.user_id, "Delivery", f"Delivered order {self.current_order_id}", prompt)
        self.current_order_id = None
        with lock:
            delivery_response_times.append(rt)
        return rt

def customer_thread(customer):
    try:
        start_time = time.time()
        while time.time() - start_time < TEST_DURATION:
            customer.place_order()
            while time.time() - start_time < TEST_DURATION:
                customer.check_status()
                time.sleep(INTERVAL_CUSTOMER)
    except Exception as e:
        log_file.write(f"Exception in customer_thread {customer.user_id}: {e}\n")
        log_file.flush()

def delivery_thread(dp):
    try:
        dp.start_shift()
        start_time = time.time()
        while time.time() - start_time < TEST_DURATION:
            dp.get_next_order()
            if dp.current_order_id:
                time.sleep(5)
                dp.deliver_order()
            time.sleep(INTERVAL_DELIVERY)
    except Exception as e:
        log_file.write(f"Exception in delivery_thread {dp.user_id}: {e}\n")
        log_file.flush()

def monitor_resources():
    end_time = time.time() + TEST_DURATION
    while time.time() < end_time:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        with lock:
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)
        time.sleep(1)

def monitor_tokens():
    global tokens_per_second
    last_total = 0
    while True:
        time.sleep(1)
        with token_lock:
            current_total = total_tokens
        tokens_per_second = current_total - last_total
        last_total = current_total

def generate_metrics_table():
    with lock:
        metrics = {
            "Customer Actions": len(customer_response_times),
            "Avg Customer RT": f"{statistics.mean(customer_response_times):.2f} s" if customer_response_times else "0.00 s",
            "Delivery Actions": len(delivery_response_times),
            "Avg Delivery RT": f"{statistics.mean(delivery_response_times):.2f} s" if delivery_response_times else "0.00 s",
            "Active Requests": len(active_requests),
            "CPU Usage": f"{cpu_usages[-1]:.2f}%" if cpu_usages else "0.00%",
            "Memory Usage": f"{memory_usages[-1]:.2f}%" if memory_usages else "0.00%",
            "Tokens Per Second": tokens_per_second
        }
    table = Table(show_header=False, box=box.MINIMAL)
    table.add_column(style="cyan")
    table.add_column(style="magenta")
    for key, value in metrics.items():
        table.add_row(key, str(value))
    return table

def generate_active_requests_table():
    with lock:
        requests = list(active_requests.items())
    if requests:
        table = Table(show_lines=True)
        table.add_column("User", style="cyan", width=12)
        table.add_column("Action", style="white", width=15, overflow="fold")
        table.add_column("Response", style="green", overflow="fold")
        for user_id, [user_type, action, response_text] in requests:
            response_str = response_text[-100:] if len(response_text) > 100 else response_text
            table.add_row(f"{user_type} {user_id}", action, response_str)
        return table
    return Text("No active requests", style="yellow")

def generate_logs_table(max_rows):
    with lock:
        recent_logs = logs[-max_rows:]
    table = Table(box=box.MINIMAL)
    table.add_column("Timestamp", style="cyan")
    table.add_column("User", style="magenta")
    table.add_column("Action", style="white", overflow="fold")
    for log in recent_logs:
        table.add_row(log["timestamp"], log["user_id"], log["action"])
    return table

if __name__ == "__main__":
    customers = [Customer(i) for i in range(1, NUM_CUSTOMERS + 1)]
    delivery_personnel = [DeliveryPersonnel(i) for i in range(11, 11 + NUM_DELIVERY)]
    threads = [
        threading.Thread(target=monitor_resources),
        threading.Thread(target=monitor_tokens)
    ] + [threading.Thread(target=customer_thread, args=(c,)) for c in customers] + \
      [threading.Thread(target=delivery_thread, args=(dp,)) for dp in delivery_personnel]
    
    for t in threads:
        t.start()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=12),
        Layout(name="main", ratio=1),
        Layout(name="progress", size=3)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="active_requests", ratio=3)
    )
    layout["left"].split_column(
        Layout(name="metrics", size=12),
        Layout(name="logs", ratio=1)
    )
    progress = Progress()
    task = progress.add_task("Test progress", total=TEST_DURATION)
    config_text = Text(justify="center")
    config_text.append("OLLAMA STRESS TESTER\n\n", style="bold magenta")
    config_text.append(f"Model: {MODEL_NAME}\nOllama URL: {OLLAMA_URL}\n")
    config_text.append(f"Customers: {NUM_CUSTOMERS}\nDelivery Personnel: {NUM_DELIVERY}\n")
    config_text.append(f"Duration: {TEST_DURATION}s\nCustomer Interval: {INTERVAL_CUSTOMER}s\nDelivery Interval: {INTERVAL_DELIVERY}s\n")
    title_panel = Panel(config_text, border_style="bright_blue")

    with Live(layout, refresh_per_second=1, console=console):
        start_time = time.time()
        while time.time() - start_time < TEST_DURATION:
            layout["header"].update(title_panel)
            layout["metrics"].update(Panel(generate_metrics_table(), title="Metrics", border_style="green"))
            layout["active_requests"].update(Panel(generate_active_requests_table(), title="Active Requests", border_style="blue"))
            layout["logs"].update(Panel(generate_logs_table(console.size.height - 20), title="Logs", border_style="yellow"))
            progress.update(task, completed=time.time() - start_time)
            layout["progress"].update(progress)
            time.sleep(1)

    for t in threads:
        t.join()

    print("Rapport de Test de Charge LLM")
    print("-----------------------------")
    print(f"Nombre de clients simulés : {NUM_CUSTOMERS}")
    print(f"Nombre de livreurs simulés : {NUM_DELIVERY}")
    print(f"Durée du test : {TEST_DURATION} secondes")
    print(f"Temps de réponse moyen des clients : {statistics.mean(customer_response_times):.2f} s" if customer_response_times else "0.00 s")
    print(f"Temps de réponse maximal des clients : {max(customer_response_times):.2f} s" if customer_response_times else "0.00 s")
    print(f"Temps de réponse moyen des livreurs : {statistics.mean(delivery_response_times):.2f} s" if delivery_response_times else "0.00 s")
    print(f"Temps de réponse maximal des livreurs : {max(delivery_response_times):.2f} s" if delivery_response_times else "0.00 s")
    print(f"Utilisation moyenne du CPU : {statistics.mean(cpu_usages):.2f}%" if cpu_usages else "0.00%")
    print(f"Utilisation maximale du CPU : {max(cpu_usages):.2f}%" if cpu_usages else "0.00%")
    print(f"Utilisation moyenne de la mémoire : {statistics.mean(memory_usages):.2f}%" if memory_usages else "0.00%")
    print(f"Utilisation maximale de la mémoire : {max(memory_usages):.2f}%" if memory_usages else "0.00%")
    log_file.close()
