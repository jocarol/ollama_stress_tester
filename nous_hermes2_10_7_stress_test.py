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
    """Send a streaming prompt to the model, update active_requests with ongoing response, and return the full response and response time."""
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
        end_time = time.time()
        duration = end_time - start_time
        log_file.write(f"Request for user {user_id} completed in {duration:.2f} seconds with {chunk_count} chunks. Full response: {accumulated_text}\n")
        log_file.flush()
        return accumulated_text, duration
    except requests.exceptions.Timeout:
        log_file.write(f"Request for user {user_id} timed out after 600 seconds\n")
        log_file.flush()
        with lock:
            if user_id in active_requests:
                user_type = active_requests[user_id][0]
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"{user_type} {user_id}",
                    "action": "Request timed out"
                })
        return "Timeout", 600
    except Exception as e:
        log_file.write(f"Error during request for user {user_id}: {e}\n")
        log_file.flush()
        with lock:
            if user_id in active_requests:
                user_type = active_requests[user_id][0]
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"{user_type} {user_id}",
                    "action": f"Request failed: {str(e)}"
                })
        return "Error", 0

class Customer:
    def __init__(self, user_id):
        self.user_id = user_id
        self.order_id = None

    def place_order(self):
        with lock:
            active_requests[self.user_id] = ["Customer", "Placing Order", ""]
        try:
            prompt = "En tant que client, je veux passer une commande pour une pizza."
            response, rt = send_prompt(self.user_id, prompt)
            response = response.strip()
            log_file.write(f"--- Raw response for Customer {self.user_id}: '{response}' ---\n")
            log_file.flush()
            match = re.search(r"Votre numéro de commande est (\d+).", response)
            if match:
                self.order_id = match.group(1)
                with lock:
                    logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "user_id": f"Customer {self.user_id}",
                        "action": f"Placed order {self.order_id}"
                    })
            else:
                with lock:
                    logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "user_id": f"Customer {self.user_id}",
                        "action": f"Failed to place order: Response did not match expected format. Got: '{response}'"
                    })
            with lock:
                customer_response_times.append(rt)
            return rt
        finally:
            with lock:
                if self.user_id in active_requests:
                    del active_requests[self.user_id]

    def check_status(self):
        if self.order_id is None:
            return 0
        with lock:
            active_requests[self.user_id] = ["Customer", "Checking Status", ""]
        try:
            prompt = f"En tant que client, où en est ma commande {self.order_id} ?"
            response, rt = send_prompt(self.user_id, prompt)
            response = response.strip()
            log_file.write(f"--- Raw response for Customer {self.user_id}: '{response}' ---\n")
            log_file.flush()
            with lock:
                customer_response_times.append(rt)
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"Customer {self.user_id}",
                    "action": f"Checked status of order {self.order_id}"
                })
            return rt
        finally:
            with lock:
                if self.user_id in active_requests:
                    del active_requests[self.user_id]

class DeliveryPersonnel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.current_order_id = None

    def start_shift(self):
        with lock:
            active_requests[self.user_id] = ["Delivery", "Starting Shift", ""]
        try:
            prompt = "En tant que livreur, je veux commencer mon quart."
            response, rt = send_prompt(self.user_id, prompt)
            response = response.strip()
            log_file.write(f"--- Raw response for Delivery {self.user_id}: '{response}' ---\n")
            log_file.flush()
            with lock:
                delivery_response_times.append(rt)
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"Delivery {self.user_id}",
                    "action": "Started shift"
                })
            return rt
        finally:
            with lock:
                if self.user_id in active_requests:
                    del active_requests[self.user_id]

    def get_next_order(self):
        with lock:
            active_requests[self.user_id] = ["Delivery", "Getting Next Order", ""]
        try:
            prompt = "En tant que livreur, je suis prêt pour la prochaine commande."
            response, rt = send_prompt(self.user_id, prompt)
            response = response.strip()
            log_file.write(f"--- Raw response for Delivery {self.user_id}: '{response}' ---\n")
            log_file.flush()
            match = re.search(r"Vous avez une nouvelle commande\s*:\s*(\d+)", response, re.IGNORECASE)
            if match:
                self.current_order_id = match.group(1)
                with lock:
                    logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "user_id": f"Delivery {self.user_id}",
                        "action": f"Got next order {self.current_order_id}"
                    })
            else:
                if "Aucune commande disponible pour le moment." in response:
                    with lock:
                        logs.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "user_id": f"Delivery {self.user_id}",
                            "action": "No orders available at the moment"
                        })
                else:
                    with lock:
                        logs.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "user_id": f"Delivery {self.user_id}",
                            "action": f"Failed to get next order: Response did not match expected format. Got: '{response}'"
                        })
            with lock:
                delivery_response_times.append(rt)
            return rt
        finally:
            with lock:
                if self.user_id in active_requests:
                    del active_requests[self.user_id]

    def deliver_order(self):
        if self.current_order_id is None:
            return 0
        with lock:
            active_requests[self.user_id] = ["Delivery", "Delivering Order", ""]
        try:
            prompt = f"En tant que livreur, j'ai livré la commande {self.current_order_id}."
            response, rt = send_prompt(self.user_id, prompt)
            response = response.strip()
            log_file.write(f"--- Raw response for Delivery {self.user_id}: '{response}' ---\n")
            log_file.flush()
            with lock:
                delivery_response_times.append(rt)
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_id": f"Delivery {self.user_id}",
                    "action": f"Delivered order {self.current_order_id}"
                })
            self.current_order_id = None
            return rt
        finally:
            with lock:
                if self.user_id in active_requests:
                    del active_requests[self.user_id]

def customer_thread(customer):
    start_time = time.time()
    while time.time() - start_time < TEST_DURATION:
        if time.time() - start_time >= TEST_DURATION:
            break
        customer.place_order()
        while time.time() - start_time < TEST_DURATION:
            if time.time() - start_time >= TEST_DURATION:
                break
            customer.check_status()
            time.sleep(INTERVAL_CUSTOMER)

def delivery_thread(dp):
    dp.start_shift()
    start_time = time.time()
    while time.time() - start_time < TEST_DURATION:
        dp.get_next_order()
        if dp.current_order_id:
            time.sleep(5)
            dp.deliver_order()
        time.sleep(INTERVAL_DELIVERY)

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
        tokens_in_last_second = current_total - last_total
        last_total = current_total
        tokens_per_second = tokens_in_last_second
        log_file.write(f"Tokens in last second: {tokens_in_last_second}, Total tokens: {current_total}\n")
        log_file.flush()

def generate_metrics_table():
    with lock:
        customer_actions = len(customer_response_times)
        delivery_actions = len(delivery_response_times)
        avg_customer_rt = statistics.mean(customer_response_times) if customer_response_times else 0
        avg_delivery_rt = statistics.mean(delivery_response_times) if delivery_response_times else 0
        current_active = len(active_requests)
        cu = cpu_usages[-1] if cpu_usages else 0
        mu = memory_usages[-1] if memory_usages else 0
    table = Table(show_header=False, box=box.MINIMAL)
    table.add_column(style="cyan")
    table.add_column(style="magenta")
    table.add_row("Customer Actions", str(customer_actions))
    table.add_row("Avg Customer RT", f"{avg_customer_rt:.2f} s")
    table.add_row("Delivery Actions", str(delivery_actions))
    table.add_row("Avg Delivery RT", f"{avg_delivery_rt:.2f} s")
    table.add_row("Active Requests", str(current_active))
    table.add_row("Queued Requests", str(current_active))
    table.add_row("CPU Usage", f"{cu:.2f}%")
    table.add_row("Memory Usage", f"{mu:.2f}%")
    table.add_row("Tokens Per Second", str(tokens_per_second))
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
            user_str = f"{user_type} {user_id}"
            action_str = action
            response_str = response_text
            table.add_row(user_str, action_str, response_str)
        return table
    else:
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
    resource_monitor = threading.Thread(target=monitor_resources)
    token_monitor = threading.Thread(target=monitor_tokens)
    resource_monitor.start()
    token_monitor.start()
    customer_threads = []
    for customer in customers:
        t = threading.Thread(target=customer_thread, args=(customer,))
        t.start()
        customer_threads.append(t)
    delivery_threads = []
    for dp in delivery_personnel:
        t = threading.Thread(target=delivery_thread, args=(dp,))
        t.start()
        delivery_threads.append(t)
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
    config_text.append("Model: ", style="bold cyan")
    config_text.append(f"{MODEL_NAME}\n", style="green")
    config_text.append("Ollama URL: ", style="bold cyan")
    config_text.append(f"{OLLAMA_URL}\n", style="green")
    config_text.append("Number of Customers: ", style="bold cyan")
    config_text.append(f"{NUM_CUSTOMERS}\n", style="green")
    config_text.append("Number of Delivery Personnel: ", style="bold cyan")
    config_text.append(f"{NUM_DELIVERY}\n", style="green")
    config_text.append("Test Duration: ", style="bold cyan")
    config_text.append(f"{TEST_DURATION} seconds\n", style="green")
    config_text.append("Customer Interval: ", style="bold cyan")
    config_text.append(f"{INTERVAL_CUSTOMER} seconds\n", style="green")
    config_text.append("Delivery Interval: ", style="bold cyan")
    config_text.append(f"{INTERVAL_DELIVERY} seconds\n", style="green")
    title_panel = Panel(config_text, border_style="bright_blue")
    with Live(layout, refresh_per_second=1, console=console):
        start_time = time.time()
        while time.time() - start_time < TEST_DURATION:
            layout["header"].update(title_panel)
            metrics_table = generate_metrics_table()
            active_table = generate_active_requests_table()  # Define active_table here
            term_height = console.size.height
            max_rows = max(1, term_height - 20)
            logs_table = generate_logs_table(max_rows)
            layout["metrics"].update(Panel(metrics_table, title="Metrics", border_style="green"))
            layout["active_requests"].update(Panel(active_table, title="Active Requests", border_style="blue", padding=0))
            layout["logs"].update(Panel(logs_table, title="Logs", border_style="yellow"))
            elapsed = time.time() - start_time
            progress.update(task, completed=elapsed)
            layout["progress"].update(progress)
            time.sleep(1)
    for t in customer_threads + delivery_threads + [resource_monitor, token_monitor]:
        t.join()
    avg_customer_rt = statistics.mean(customer_response_times) if customer_response_times else 0
    max_customer_rt = max(customer_response_times) if customer_response_times else 0
    avg_delivery_rt = statistics.mean(delivery_response_times) if delivery_response_times else 0
    max_delivery_rt = max(delivery_response_times) if delivery_response_times else 0
    avg_cpu = statistics.mean(cpu_usages) if cpu_usages else 0
    max_cpu = max(cpu_usages) if cpu_usages else 0
    avg_memory = statistics.mean(memory_usages) if memory_usages else 0
    max_memory = max(memory_usages) if memory_usages else 0
    print("Rapport de Test de Charge LLM")
    print("-----------------------------")
    print(f"Nombre de clients simulés : {NUM_CUSTOMERS}")
    print(f"Nombre de livreurs simulés : {NUM_DELIVERY}")
    print(f"Durée du test : {TEST_DURATION} secondes")
    print(f"Temps de réponse moyen des clients : {avg_customer_rt:.2f} s")
    print(f"Temps de réponse maximal des clients : {max_customer_rt:.2f} s")
    print(f"Temps de réponse moyen des livreurs : {avg_delivery_rt:.2f} s")
    print(f"Temps de réponse maximal des livreurs : {max_delivery_rt:.2f} s")
    print(f"Utilisation moyenne du CPU : {avg_cpu:.2f}%")
    print(f"Utilisation maximale du CPU : {max_cpu:.2f}%")
    print(f"Utilisation moyenne de la mémoire : {avg_memory:.2f}%")
    print(f"Utilisation maximale de la mémoire : {max_memory:.2f}%")
    log_file.close()
