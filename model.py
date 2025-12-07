import shift
import time
import math
from datetime import datetime, timedelta

gamma = 0.01  # risk aversion γ
sigma = 0.50  # volatility σ
kappa = 1.5  # market depth κ

phi_max = 2  # AS inventory-size response, lots (1 lot = 100 shares)
eta = -0.005  # AS exponential inventory penalty (paper)

SYMBOLS = ["AAPL", "AMZN", "BRKb", "CS1", "CS2", "GOOG", "MSFT"]

TRADE_DURATION_MINUTES = 1.0
TRADE_DURATION_SECONDS = TRADE_DURATION_MINUTES * 60.0

UPDATE_INTERVAL = 1.0
WAIT_TIME = 5.0

def safe_get_inventory(trader, sym):
    try:
        return trader.get_portfolio_item(sym).get_shares()
    except:
        return 0


def safe_get_realized_pl(trader, sym):
    try:
        return trader.get_portfolio_item(sym).get_realized_pl()
    except:
        return 0.0


def get_mid_price(best):
    return (best.get_bid_price() + best.get_ask_price()) / 2.0


def compute_dynamic_size(q):
    """
    φ_bid, φ_ask inventory function FROM THE PAPER.
    q = inventory in shares.
    φ_max = max lots.
    """
    if q < 0:  # short → buy aggressively
        phi_bid = phi_max
        phi_ask = phi_max * math.exp(-eta * q)
    else:  # long → sell aggressively
        phi_bid = phi_max * math.exp(-eta * q)
        phi_ask = phi_max
    return max(1, int(phi_bid)), max(1, int(phi_ask))


def compute_as_quotes(mid, q, gamma, sigma, kappa, T_minus_t):
    """
    Avellaneda–Stoikov quoting

    r_t = s_t − q_t γ σ² (T − t)
    δ_a + δ_b = γ σ² (T − t) + 2 ln(1 + γ/κ)
    δ = (δ_a + δ_b)/2
    bid = r_t − δ
    ask = r_t + δ
    """

    # Indifference price
    r = mid - q * gamma * (sigma ** 2) * T_minus_t

    # Optimal total spread
    spread = gamma * (sigma ** 2) * T_minus_t + 2 * math.log(1 + gamma / kappa)

    delta = spread / 2.0

    bid = r - delta
    ask = r + delta

    # Round to tick (0.01)
    bid = round(bid, 2)
    ask = round(ask, 2)

    # enforce no-cross
    if bid >= ask:
        bid = ask - 0.01

    # enforce positivity
    bid = max(0.01, bid)

    return bid, ask

def main():
    trader = shift.Trader("ksingh29")
    trader.connect("initiator.cfg", "o8WS6RAN")
    print("Connected:", trader.is_connected())

    start_time = time.time()
    end_time = start_time + TRADE_DURATION_SECONDS

    # Per-symbol state
    state = {}
    for sym in SYMBOLS:
        state[sym] = {
            "initial_pl": safe_get_realized_pl(trader, sym),
            "last_time": time.time(),
            "trades": [],
            "max_long": 0,
            "max_short": 0,
            "max_abs": 0,
        }

    processed_exec_ids = set()
    loop_count = 0

    # =================== MAIN LOOP =====================
    while time.time() < end_time:

        loop_count += 1
        print(f"\n============= LOOP {loop_count} =============")

        t_elapsed = time.time() - start_time
        T_minus_t = max(0, TRADE_DURATION_SECONDS - t_elapsed)

        for sym in SYMBOLS:

            print(f"\n--- {sym} ---")

            best = trader.get_best_price(sym)
            mid = get_mid_price(best)
            q = safe_get_inventory(trader, sym)

            print(f"Mid: {mid:.4f} | Inventory: {q}")

            # Track inventory extremes
            state[sym]["max_long"] = max(state[sym]["max_long"], q)
            state[sym]["max_short"] = min(state[sym]["max_short"], q)
            state[sym]["max_abs"] = max(state[sym]["max_abs"], abs(q))

            size_bid, size_ask = compute_dynamic_size(q)
            size_bid = max(1, min(size_bid, 5))
            size_ask = max(1, min(size_ask, 5))

            # ***** PURE PAPER QUOTES *****
            bid_px, ask_px = compute_as_quotes(
                mid=mid,
                q=q,
                gamma=gamma,
                sigma=sigma,
                kappa=kappa,
                T_minus_t=T_minus_t
            )

            print(f"AS Quotes: BID={bid_px:.4f}  ASK={ask_px:.4f} | sizes: {size_bid}/{size_ask}")

            # Algorithm 1 logic
            waiting_list_sym = [o for o in trader.get_waiting_list() if o.symbol == sym]
            waiting_size = len(waiting_list_sym)

            print(f"Waiting orders ({sym}): {waiting_size}")
            last_time = state[sym]["last_time"]

            # 1) No orders resting → place both sides
            if waiting_size == 0:
                print("Action: place fresh AS quotes")
                trader.submit_order(shift.Order(shift.Order.Type.LIMIT_BUY, sym, size_bid, bid_px))
                trader.submit_order(shift.Order(shift.Order.Type.LIMIT_SELL, sym, size_ask, ask_px))

                state[sym]["last_time"] = time.time()

            # 2) One-sided fill → wait then refresh
            elif waiting_size == 1:
                if time.time() - last_time > WAIT_TIME:
                    print("Action: timeout → cancel & requote")
                    for o in waiting_list_sym:
                        trader.submit_cancellation(o)

                    time.sleep(0.2)
                    trader.submit_order(shift.Order(shift.Order.Type.LIMIT_BUY, sym, size_bid, bid_px))
                    trader.submit_order(shift.Order(shift.Order.Type.LIMIT_SELL, sym, size_ask, ask_px))

                    state[sym]["last_time"] = time.time()

            # 3) Two resting → periodic refresh
            else:
                if time.time() - last_time > UPDATE_INTERVAL:
                    print("Action: periodic refresh")
                    for o in waiting_list_sym:
                        trader.submit_cancellation(o)
                    time.sleep(0.2)

                    trader.submit_order(shift.Order(shift.Order.Type.LIMIT_BUY, sym, size_bid, bid_px))
                    trader.submit_order(shift.Order(shift.Order.Type.LIMIT_SELL, sym, size_ask, ask_px))

                    state[sym]["last_time"] = time.time()

        # ---------------- EXECUTION PROCESSING ----------------
        for order in trader.get_submitted_orders():
            if order.executed_size > 0 and order.id not in processed_exec_ids:

                execs = trader.get_executed_orders(order.id)
                for ex in execs:
                    if ex.executed_size <= 0:
                        continue

                    sym = ex.symbol
                    if sym not in state:
                        continue

                    side = "BUY" if "BUY" in str(ex.type) else "SELL"
                    shares = ex.executed_size * 100
                    px = ex.executed_price

                    print(f"EXECUTED {sym}: {shares} @ {px} ({side})")

                    state[sym]["trades"].append(
                        {"side": side, "qty": shares, "price": px}
                    )

                processed_exec_ids.add(order.id)

        time.sleep(0.1)


    print("\n================================================")
    print("                FINAL PERFORMANCE REPORT         ")
    print("================================================\n")

    total_net_pl = 0.0

    for sym in SYMBOLS:

        print(f"\n----- {sym} -----")

        final_pl = safe_get_realized_pl(trader, sym)
        net_pl = final_pl - state[sym]["initial_pl"]
        total_net_pl += net_pl

        print(f"Initial P&L:          {state[sym]['initial_pl']:.2f}")
        print(f"Final P&L:            {final_pl:.2f}")
        print(f"Net P&L:              {net_pl:.2f}")

        print(f"Ending Inventory:     {safe_get_inventory(trader, sym)}")
        print(f"Max Long Inventory:    {state[sym]['max_long']}")
        print(f"Max Short Inventory:   {state[sym]['max_short']}")
        print(f"Max Absolute Inventory:{state[sym]['max_abs']}")

        trades = state[sym]["trades"]
        print(f"Total Executions:     {len(trades)}")

        if trades:
            buys = [t for t in trades if t["side"] == "BUY"]
            sells = [t for t in trades if t["side"] == "SELL"]

            def weighted_avg(trs):
                qty = sum(t["qty"] for t in trs)
                if qty == 0: return 0
                return sum(t["qty"] * t["price"] for t in trs) / qty

            avg_buy = weighted_avg(buys)
            avg_sell = weighted_avg(sells)

            profitable = sum(
                1 for t in sells if t["price"] > avg_buy and avg_buy > 0
            )
            winrate = profitable / len(trades)

            print(f"Buys:                 {len(buys)}")
            print(f"Sells:                {len(sells)}")
            print(f"Avg Buy Price:        {avg_buy:.4f}")
            print(f"Avg Sell Price:       {avg_sell:.4f}")
            print(f"Win Rate:             {winrate * 100:.2f}%")
        else:
            print("No executions recorded.")

    print("\n================================================")
    print(f"TOTAL STRATEGY NET P&L: {total_net_pl:.2f}")
    print("================================================\n")

    trader.disconnect()
    print("DONE.")


if __name__ == "__main__":
    main()
