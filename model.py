import shift
from time import sleep
from datetime import timedelta
from threading import Thread, Lock

import numpy as np
from arch import arch_model
from collections import deque

###############################################################################
# ------------------------ CONFIG / PARAMETERS -------------------------------
###############################################################################

SYMBOLS = ["AAPL", "AMZN", "BRKb", "GOOG", "MSFT"]
TRADE_DURATION_MINUTES = 1        # <----- YOU CONTROL THIS

PRICE_WINDOW = 120                # midprice window for GARCH
GARCH_UPDATE_EVERY = 5

gamma = 0.01
kappa = 1.5
phi_max = 2
eta = -0.005

###############################################################################
# ------------------------ GLOBAL STATE FOR REPORTING ------------------------
###############################################################################

# store all per-symbol results for final report
thread_state = {
    sym: {
        "initial_pl": 0.0,
        "trades": [],
        "max_long": 0,
        "max_short": 0,
        "max_abs": 0,
    }
    for sym in SYMBOLS
}

state_lock = Lock()   # thread-safe writes

rolling_mids = {sym: deque(maxlen=PRICE_WINDOW) for sym in SYMBOLS}
current_sigma = {sym: 0.50 for sym in SYMBOLS}
last_valid_mid = {sym: None for sym in SYMBOLS}


###############################################################################
# ------------------------ HELPER FUNCTIONS ----------------------------------
###############################################################################

def get_safe_mid(best, sym):
    bid = best.get_bid_price()
    ask = best.get_ask_price()
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        last_valid_mid[sym] = mid
        return mid
    if bid > 0:
        last_valid_mid[sym] = bid
        return bid
    if ask > 0:
        last_valid_mid[sym] = ask
        return ask
    return last_valid_mid[sym] if last_valid_mid[sym] else 0.01


def estimate_sigma_garch(returns):
    returns = np.array(returns)
    returns = returns[np.abs(returns) > 1e-12]

    if len(returns) < 10:
        return 0.50

    scaled = returns * 100.0
    try:
        am = arch_model(scaled, vol="Garch", p=1, q=1,
                        mean="Zero", dist="normal", rescale=False)
        res = am.fit(disp="off")
        var = res.forecast(horizon=1).variance.iloc[-1, 0]
        sigma = float(np.sqrt(var)) / 100.0
        return max(1e-4, min(sigma, 1.0))
    except:
        return max(1e-4, min(returns.std(), 1.0))


def compute_dynamic_size(q):
    if q < 0:
        phi_bid = phi_max
        phi_ask = phi_max * np.exp(-eta * q)
    else:
        phi_bid = phi_max * np.exp(-eta * q)
        phi_ask = phi_max
    return min(max(1, int(phi_bid)), 5), min(max(1, int(phi_ask)), 5)


def compute_as_quotes(mid, q, sigma, gamma, kappa, T_minus_t):
    r = mid - q * gamma * (sigma ** 2) * T_minus_t
    spread = gamma * (sigma ** 2) * T_minus_t + 2 * np.log(1 + gamma / kappa)
    delta = spread / 2.0
    bid = round(r - delta, 2)
    ask = round(r + delta, 2)
    if bid >= ask:
        bid = ask - 0.01
    return max(0.01, bid), ask


def cancel_orders(trader, ticker):
    for o in trader.get_waiting_list():
        if o.symbol == ticker:
            trader.submit_cancellation(o)
            sleep(0.1)


def close_positions(trader, ticker):
    item = trader.get_portfolio_item(ticker)
    long_shares = item.get_long_shares()
    short_shares = item.get_short_shares()
    if long_shares > 0:
        trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, ticker, long_shares // 100))
    if short_shares > 0:
        trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, ticker, short_shares // 100))
    sleep(0.5)


###############################################################################
# ------------------------ STRATEGY THREAD -----------------------------------
###############################################################################

def strategy(trader, ticker, endtime):
    print(f"[START] Strategy for {ticker}")

    # initialize P&L baseline in global state
    thread_state[ticker]["initial_pl"] = trader.get_portfolio_item(ticker).get_realized_pl()

    best = trader.get_best_price(ticker)
    prev_mid = get_safe_mid(best, ticker)

    while trader.get_last_trade_time() < endtime:

        cancel_orders(trader, ticker)

        best = trader.get_best_price(ticker)
        mid = get_safe_mid(best, ticker)

        rolling_mids[ticker].append(mid)

        mids = np.array(rolling_mids[ticker])
        logp = np.log(mids[mids > 0]) if len(mids) > 1 else []
        rets = np.diff(logp)

        # update volatility
        if len(rets) >= 5:
            current_sigma[ticker] = estimate_sigma_garch(rets)

        sigma = current_sigma[ticker]
        q = trader.get_portfolio_item(ticker).get_shares()

        # track extremes (thread-safe)
        with state_lock:
            thread_state[ticker]["max_long"] = max(thread_state[ticker]["max_long"], q)
            thread_state[ticker]["max_short"] = min(thread_state[ticker]["max_short"], q)
            thread_state[ticker]["max_abs"] = max(thread_state[ticker]["max_abs"], abs(q))

        T_minus_t = 30  # fixed small horizon
        bid, ask = compute_as_quotes(mid, q, sigma, gamma, kappa, T_minus_t)
        size_bid, size_ask = compute_dynamic_size(q)

        # submit pair quotes
        trader.submit_order(shift.Order(shift.Order.Type.LIMIT_BUY, ticker, size_bid, bid))
        trader.submit_order(shift.Order(shift.Order.Type.LIMIT_SELL, ticker, size_ask, ask))

        prev_mid = mid
        sleep(0.5)

    # shutdown:
    cancel_orders(trader, ticker)
    close_positions(trader, ticker)

    # record all executions for final report
    submitted = trader.get_submitted_orders()
    for order in submitted:
        execs = trader.get_executed_orders(order.id)
        for ex in execs:
            if ex.executed_size > 0:
                side = "BUY" if "BUY" in str(ex.type) else "SELL"
                qty = ex.executed_size * 100
                px = ex.executed_price
                with state_lock:
                    thread_state[ticker]["trades"].append(
                        {"side": side, "qty": qty, "price": px}
                    )

    print(f"[END] Strategy for {ticker}")


###############################################################################
# ------------------------ FINAL REPORT --------------------------------------
###############################################################################

def final_report(trader):
    print("\n================================================")
    print("              FINAL PERFORMANCE REPORT")
    print("================================================\n")

    total_net = 0

    for sym in SYMBOLS:
        print(f"\n----- {sym} -----")

        initial = thread_state[sym]["initial_pl"]
        final_pl = trader.get_portfolio_item(sym).get_realized_pl()
        net = final_pl - initial
        total_net += net

        trades = thread_state[sym]["trades"]

        print(f"Initial P&L:         {initial:.2f}")
        print(f"Final P&L:           {final_pl:.2f}")
        print(f"Net P&L:             {net:.2f}")
        print(f"Max Long Inventory:  {thread_state[sym]['max_long']}")
        print(f"Max Short Inventory: {thread_state[sym]['max_short']}")
        print(f"Max Abs Inventory:   {thread_state[sym]['max_abs']}")
        print(f"Total Executions:    {len(trades)}")

    print("\n================================================")
    print(f"TOTAL STRATEGY NET P&L: {total_net:.2f}")
    print("================================================\n")


###############################################################################
# ------------------------ MAIN (SHIFT LEADERBOARD) --------------------------
###############################################################################

def main(trader):
    current_time = trader.get_last_trade_time()
    start_time = current_time
    end_time = start_time + timedelta(minutes=TRADE_DURATION_MINUTES)

    print(f"[CONFIG] Trading for {TRADE_DURATION_MINUTES} minutes")

    threads = [
        Thread(target=strategy, args=(trader, sym, end_time))
        for sym in SYMBOLS
    ]

    # start threads
    for t in threads:
        t.start()
        sleep(0.3)

    # wait for end time
    while trader.get_last_trade_time() < end_time:
        sleep(1)

    # wait for strategies to finish
    for t in threads:
        t.join()

    # ensure positions closed
    for sym in SYMBOLS:
        cancel_orders(trader, sym)
        close_positions(trader, sym)

    final_report(trader)


###############################################################################
# ------------------------ ENTRYPOINT ----------------------------------------
###############################################################################

if __name__ == "__main__":
    with shift.Trader("ksingh29") as trader:
        trader.connect("initiator.cfg", "o8WS6RAN")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        main(trader)
