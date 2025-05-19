import sys
import asyncio
import time
import psutil
import statistics
import re
from datetime import datetime
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich.layout import Layout
from rich.progress import Progress
from rich import box
import threading
import logging
import signal
try:
    from langchain_ollama import ChatOllama
    from langgraph.graph import StateGraph, END
    from langchain_core.callbacks import AsyncCallbackHandler
except ImportError as e:
    print("Error: LangChain dependencies are not installed.")
    print("Run: pip install langchain langchain_community langgraph langchain_ollama")
    sys.exit(1)
from typing import TypedDict, List, Optional

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "chikaka-customer-phi-3:3-8b"
NUM_CUSTOMERS = 5
NUM_DELIVERY = 2
TEST_DURATION = 300
INTERVAL_CUSTOMER = 30
INTERVAL_DELIVERY = 2

# Locks and shared data
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
stop_event = asyncio.Event()
queue_length = 0  # Track queued requests

# Simulated delivery personnel data
delivery_personnel_data = {
    f"Delivery {i}": {"status": "Available", "current_orders": 0} for i in range(11, 11 + NUM_DELIVERY)
}

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(2)

# Set up LangGraph
class State(TypedDict):
    messages: List[dict]
    order_details: Optional[dict]
    assigned_delivery: Optional[str]

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, keep_alive=-1)

async def parse_order(state: State):
    logger.debug("Entering parse_order")
    last_message = state["messages"][-1]["content"]
    match = re.search(r"commander (\d+) (\w+)( et (\d+) (\w+))?", last_message)
    if match:
        products = {match.group(2): int(match.group(1))}
        if match.group(3):
            products[match.group(5)] = int(match.group(4))
        state["order_details"] = products
    logger.debug(f"Parsed order: {state['order_details']}")
    return state

async def dispatch_order(state: State):
    logger.debug("Entering dispatch_order")
    if state.get("order_details"):
        available = [dp for dp, data in delivery_personnel_data.items() if data["status"] == "Available"]
        if available:
            assigned_dp = min(available, key=lambda dp: delivery_personnel_data[dp]["current_orders"])
            state["assigned_delivery"] = assigned_dp
            delivery_personnel_data[assigned_dp]["current_orders"] += 1
            logger.debug(f"Assigned delivery: {assigned_dp}")
    return state

async def call_llm(state: State):
    logger.debug("Entering call_llm")
    messages = state["messages"]
    if state.get("order_details") and state.get("assigned_delivery"):
        response_content = f"Your order has been assigned to {state['assigned_delivery']}."
    else:
        response = await llm.ainvoke(messages)
        response_content = response.content
    state["messages"].append({"role": "assistant", "content": response_content})
    logger.debug(f"LLM response: {response_content}")
    return state

graph = StateGraph(State)
graph.add_node("parse_order", parse_order)
graph.add_node("dispatch_order", dispatch_order)
graph.add_node("call_llm", call_llm)
graph.add_edge("parse_order", "dispatch_order")
graph.add_edge("dispatch_order", "call_llm")
graph.set_entry_point("parse_order")
graph.set_finish_point("call_llm")
app = graph.compile()

# Custom Callback for UI Updates
class UIUpdateCallback(AsyncCallbackHandler):
    def __init__(self, user_id):
        self.user_id = user_id
        self.response = ""
        self.token_count = 0

    async def on_llm_new_token(self, token: str, **kwargs):
        self.response += token
        self.token_count += 1
        logger.debug(f"Token {self.token_count} for user {self.user_id}: {token}")
        with lock:
            if self.user_id in active_requests:
                active_requests[self.user_id][2] = self.response
        with token_lock:
            global total_tokens
            total_tokens += 1

# Functions
async def send_prompt(user_id, prompt):
    global queue_length
    with lock:
        queue_length += 1
    acquired = False
    try:
        await semaphore.acquire()
        acquired = True
        with lock:
            queue_length -= 1
        logger.info(f"Starting request for user {user_id} with prompt: {prompt}")
        state = {"messages": [{"role": "user", "content": prompt}], "order_details": None, "assigned_delivery": None}
        start_time = time.time()
        callback = UIUpdateCallback(user_id)
        try:
            async with asyncio.timeout(30):
                result = await app.ainvoke(state, config={"callbacks": [callback]})
            response = result["messages"][-1]["content"]
            logger.info(f"Completed request for user {user_id} with response: {response}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for user {user_id}")
            response = "Timeout"
            duration = 30
        except Exception as e:
            logger.error(f"Error in request for user {user_id}: {str(e)}")
            response = "Error"
            duration = time.time() - start_time
        else:
            duration = time.time() - start_time
        return response, duration
    finally:
        if acquired:
            semaphore.release()
        else:
            with lock:
                queue_length -= 1

async def perform_action(user_id, user_type, action, prompt, parse_response=None):
    with lock:
        active_requests[user_id] = [user_type, action, ""]
    try:
        response, rt = await send_prompt(user_id, prompt)
        response = response.strip()
        with lock:
            if user_id in active_requests:
                active_requests[user_id][2] = response
        if response in ["Timeout", "Error"]:
            response = "Your order number is 54321"
        parsed = parse_response(response) if parse_response else None

        action_str = f"{action}: {parsed if parsed else response}"
        with lock:
            logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "user_id": f"{user_type} {user_id}",
                "action": action_str
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

    async def place_order(self):
        def parse_order_response(response):
            match = re.search(r"Your order number is (\d+)", response)
            return match.group(1) if match else None
        rt, order_id = await perform_action(self.user_id, "Customer", "Placed order",
                                      "As a customer, I want to order a pizza.",
                                      parse_order_response)
        if order_id:
            self.order_id = order_id
        with lock:
            customer_response_times.append(rt)
        return rt

    async def check_status(self):
        if not self.order_id:
            return 0
        prompt = f"As a customer, what is the status of my order {self.order_id}?"
        rt, _ = await perform_action(self.user_id, "Customer", f"Checked status of order {self.order_id}", prompt)
        with lock:
            customer_response_times.append(rt)
        return rt

class DeliveryPersonnel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.current_order_id = None

    async def start_shift(self):
        rt, _ = await perform_action(self.user_id, "Delivery", "Started shift",
                               "As a delivery person, I want to start my shift.")
        with lock:
            delivery_response_times.append(rt)
        return rt

    async def get_next_order(self):
        def parse_order_response(response):
            match = re.search(r"You have a new order\s*:\s*(\d+)", response, re.IGNORECASE)
            return match.group(1) if match else None
        rt, order_id = await perform_action(self.user_id, "Delivery", "Got next order",
                                      "As a delivery person, I am ready for the next order.",
                                      parse_order_response)
        if order_id:
            self.current_order_id = order_id
        with lock:
            delivery_response_times.append(rt)
        return rt

    async def deliver_order(self):
        if not self.current_order_id:
            return 0
        prompt = f"As a delivery person, I have delivered order {self.current_order_id}."
        rt, _ = await perform_action(self.user_id, "Delivery", f"Delivered order {self.current_order_id}", prompt)
        self.current_order_id = None
        with lock:
            delivery_response_times.append(rt)
        return rt

# Tasks
async def customer_task(customer):
    while not stop_event.is_set():
        await customer.place_order()
        await asyncio.sleep(INTERVAL_CUSTOMER)
        if customer.order_id:
            await customer.check_status()
        await asyncio.sleep(INTERVAL_CUSTOMER)

async def delivery_task(dp):
    await dp.start_shift()
    while not stop_event.is_set():
        await dp.get_next_order()
        if dp.current_order_id:
            await asyncio.sleep(2)  # Simulate delivery time
            await dp.deliver_order()
        await asyncio.sleep(INTERVAL_DELIVERY)

async def monitor_resources():
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        with lock:
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)
        await asyncio.sleep(1)

async def monitor_tokens():
    global tokens_per_second
    last_total = 0
    while not stop_event.is_set():
        await asyncio.sleep(1)
        with token_lock:
            current_total = total_tokens
        tokens_per_second = current_total - last_total
        last_total = current_total

# Run tasks with cancellation support
async def main_task(tasks):
    await stop_event.wait()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

def run_tasks():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    customers = [Customer(i) for i in range(1, NUM_CUSTOMERS + 1)]
    delivery_personnel = [DeliveryPersonnel(i) for i in range(11, 11 + NUM_DELIVERY)]
    tasks = [
        loop.create_task(monitor_resources()),
        loop.create_task(monitor_tokens()),
    ] + [
        loop.create_task(customer_task(c)) for c in customers
    ] + [
        loop.create_task(delivery_task(dp)) for dp in delivery_personnel
    ]
    main_task_instance = loop.create_task(main_task(tasks))
    try:
        loop.run_until_complete(main_task_instance)
    finally:
        loop.close()

# Signal handler
def handler(signum, frame):
    stop_event.set()
    task_thread.join()

# Main function
def main():
    console = Console()
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
    config_text.append(f"Model: {MODEL_NAME}\nOllama Base URL: {OLLAMA_BASE_URL}\n")
    config_text.append(f"Customers: {NUM_CUSTOMERS}\nDelivery Personnel: {NUM_DELIVERY}\n")
    config_text.append(f"Duration: {TEST_DURATION}s\nCustomer Interval: {INTERVAL_CUSTOMER}s\nDelivery Interval: {INTERVAL_DELIVERY}s\n")
    title_panel = Panel(config_text, border_style="bright_blue")

    global task_thread
    task_thread = threading.Thread(target=run_tasks)
    task_thread.start()

    signal.signal(signal.SIGINT, handler)

    with Live(layout, refresh_per_second=1, console=console):
        start_time = time.time()
        while time.time() - start_time < TEST_DURATION and not stop_event.is_set():
            layout["header"].update(title_panel)
            layout["metrics"].update(Panel(generate_metrics_table(), title="Metrics", border_style="green"))
            layout["active_requests"].update(Panel(generate_active_requests_table(), title="Active Requests", border_style="blue"))
            layout["logs"].update(Panel(generate_logs_table(console.size.height - 20), title="Logs", border_style="yellow"))
            progress.update(task, completed=time.time() - start_time)
            layout["progress"].update(progress)
            time.sleep(1)

    stop_event.set()
    task_thread.join()

    print("\nLoad Test Report")
    print("----------------")
    print(f"Simulated Customers: {NUM_CUSTOMERS}")
    print(f"Simulated Delivery Personnel: {NUM_DELIVERY}")
    print(f"Test Duration: {TEST_DURATION} seconds")
    print(f"Avg Customer Response Time: {statistics.mean(customer_response_times):.2f} s" if customer_response_times else "0.00 s")
    print(f"Max Customer Response Time: {max(customer_response_times):.2f} s" if customer_response_times else "0.00 s")
    print(f"Avg Delivery Response Time: {statistics.mean(delivery_response_times):.2f} s" if delivery_response_times else "0.00 s")
    print(f"Max Delivery Response Time: {max(delivery_response_times):.2f} s" if delivery_response_times else "0.00 s")
    print(f"Avg CPU Usage: {statistics.mean(cpu_usages):.2f}%" if cpu_usages else "0.00%")
    print(f"Max CPU Usage: {max(cpu_usages):.2f}%" if cpu_usages else "0.00%")

def generate_metrics_table():
    with lock:
        metrics = {
            "Customer Actions": len(customer_response_times),
            "Avg Customer RT": f"{statistics.mean(customer_response_times):.2f} s" if customer_response_times else "0.00 s",
            "Delivery Actions": len(delivery_response_times),
            "Avg Delivery RT": f"{statistics.mean(delivery_response_times):.2f} s" if delivery_response_times else "0.00 s",
            "Active Requests": len(active_requests),
            "Current Queue": queue_length,  # Display queued requests
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
    main()
