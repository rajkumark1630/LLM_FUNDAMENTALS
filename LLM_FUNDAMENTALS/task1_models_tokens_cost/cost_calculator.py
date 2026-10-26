
def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    """
    Prices are USD per 1 million tokens.
    """
    input_cost = (input_tokens / 1_000_000) * price_in
    output_cost = (output_tokens / 1_000_000) * price_out
    return input_cost + output_cost


if __name__ == "__main__":
    chats_per_day = 1000
    days_per_month = 30

    input_tokens_per_chat = 1500
    output_tokens_per_chat = 400

    monthly_chats = chats_per_day * days_per_month
    monthly_input = monthly_chats * input_tokens_per_chat
    monthly_output = monthly_chats * output_tokens_per_chat

    # Illustrative prices only; replace with verified model pricing.
    price_in = 0.50
    price_out = 3.00

    monthly_cost = estimate_cost(
        monthly_input,
        monthly_output,
        price_in,
        price_out,
    )

    print("Monthly chats:", monthly_chats)
    print("Monthly input tokens:", monthly_input)
    print("Monthly output tokens:", monthly_output)
    print(f"Estimated monthly cost: ${monthly_cost:.2f}")
