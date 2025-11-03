import random
import json
import numpy as np
# Updated cities list with corresponding zip code ranges for realism
cities_info = {
"New York": ("NY", [10001, 10292]),
"Los Angeles": ("CA", [90001, 91607]),
"Chicago": ("IL", [60601, 60657]),
"Houston": ("TX", [77001, 77299]),
"Phoenix": ("AZ", [85001, 85099]),
"Philadelphia": ("PA", [19019, 19190]),
"San Antonio": ("TX", [78201, 78299]),
"San Diego": ("CA", [92101, 92199]),
"Dallas": ("TX", [75201, 75398]),
"San Jose": ("CA", [95101, 95196]),
"Austin": ("TX", [78701, 78799]),
"Jacksonville": ("FL", [32099, 32290]),
"Fort Worth": ("TX", [76101, 76199]),
"Columbus": ("OH", [43085, 43299]),
"Charlotte": ("NC", [28201, 28299]),
"San Francisco": ("CA", [94102, 94188]),
"Indianapolis": ("IN", [46201, 46298]),
"Seattle": ("WA", [98101, 98199]),
"Denver": ("CO", [80201, 80299]),
"Washington": ("DC", [20001, 20599])
}

first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Yusuf", "Isabella", "Lucas", "Mia", "Mason", "Sophia", "Ethan", "Aarav", "James", "Amelia", "Lei", "Harper", "Sofia", "Evelyn", "Mohamed", "Yara", "Raj", "Fatima", "Juan", "Daiki", "Mei", "Chen", "Ivan", "Anya", "Omar"]
last_names = ["Smith", "Johnson", "Patel", "Nguyen", "Garcia", "Silva", "Kim", "Santos", "Khan", "Li", "Kovacs", "Muller", "Rossi", "Hernandez", "Sanchez", "Ito", "Johansson", "Lopez", "Gonzalez", "Ahmed", "Brown", "Davis", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee"]

# Function to generate an address with a city-realistic zip code
def generate_address():
    streets = ["Maple Drive", "Oak Street", "Pine Lane", "Elm Street", "Cedar Avenue", "Hillcrest Drive", "Willow Lane", "Sunset Drive", "River Road", "Lakeview Drive", "Main Street", "Park Avenue", "Broadway", "Elm Avenue", "Highland Drive", "Chestnut Street", "Hickory Lane", "Spruce Street", "Cedar Street", "Laurel Lane"]
    city, (state, zip_range) = random.choice(list(cities_info.items()))
    address1 = f"{random.randint(100, 999)} {random.choice(streets)}"
    address2 = f"Suite {random.randint(100, 999)}"
    zip_code = str(random.randint(zip_range[0], zip_range[1]))
    return {
    "address1": address1,
    "address2": address2,
    "city": city,
    "country": "USA",
    "province": state,
    "zip": zip_code
    }

def poisson_sample(lam, max_val):
    sample = 0
    # Use rejection sampling to ensure sample is capped at max_val
    while sample == 0 or sample >= max_val:
        sample = np.random.poisson(lam)
    return sample

def generate_payment_method_id(source): # like "card_2423"/"paypal_2423"/"gift_card_0912"
    random_id = f"{source}_{random.randint(1000000, 9999999)}"
    existing_ids = [method["id"] for user in user_profiles.values() for method in user["payment_methods"].values()]
    if random_id not in existing_ids:
        return random_id
    else:
        return generate_payment_method_id(source)

# Generate payment methods
def payment_method():
    payment_types = ["credit_card", "paypal", "gift_card"]
    count_methods = poisson_sample(1, 5)
    payment_methods = []
    existing_methods = set()
    for _ in range(count_methods):
        payment_source = random.choice(payment_types)
        if payment_source == "credit_card":
            brand = random.choice(["visa", "mastercard"])
            if ("credit_card", brand) in existing_methods:
                continue
            payment_methods.append({
            "source": "credit_card",
            "brand": brand,
            "last_four": f"{random.randint(1000, 9999)}",
            "id": generate_payment_method_id("credit_card")
            })
            existing_methods.add(("credit_card", brand))
        elif payment_source == "paypal":
            if ("paypal",) in existing_methods:
                continue
            payment_methods.append({
            "source": "paypal",
            "id": generate_payment_method_id("paypal")
            })
            existing_methods.add(("paypal",))
        else:
            if ("gift_card",) in existing_methods:
                continue
            payment_methods.append({
            "source": "gift_card",
            "balance": random.randint(0, 100),
            "id": generate_payment_method_id("gift_card")
            })
            existing_methods.add(("gift_card",))
    return payment_methods

user_profiles = {}
for i in range(1):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    email_suffix, user_id_suffix = random.sample(range(1000, 10000), 2)
    email = f"{first_name.lower()}.{last_name.lower()}{email_suffix}@example.com"
    user_id = f"{first_name.lower()}_{last_name.lower()}_{user_id_suffix}"
    # addresses = [generate_address() for _ in range(poisson_sample(1, 3))]
    payment_methods = payment_method()
    user_profiles[user_id] = {
        "name": {
            "first_name": first_name,
            "last_name": last_name
        },
        "address": generate_address(),
        "email": email,
        "payment_methods": {method['id']: method for method in payment_methods},
    }
print(user_profiles)