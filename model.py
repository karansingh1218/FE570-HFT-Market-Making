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

# Symbols to trade (each runs in its own thread)
SYMBOLS = ["AAPL", "AMZN", "BRKb", "GOOG", "MSFT"]

# How long you want to trade (in minutes)
TRADE_DURATION_MINUTES = 30

# GARCH window
PRICE_WINDOW = 120
GARCH_UPDATE_EVERY = 5  # (not strictly needed here, but easy to tweak)

# Avellaneda–Stoikov parameters
gamma = 0.01
kappa = 1.5
phi_max = 2
eta = -0.005

###############################################################################
# ------------------------ GLOBAL STATE --------------------------------------
###############################################################################

# Per-symbol rolling mids & sigma
rolling_mids = {sym: deque(maxlen=PRICE_WINDOW) for sym in SYMBOLS}
current_sigma = {sym: 0.50 for sym in SYMBOLS}
last_valid_mid = {sym: None for sym in SYMBOLS}

# Per-symbol stats for final report
thread_state = {
    sym: {
        "initial_pl": 0.0,
        "trades": [],       # list of {side, qty, price}
        "max_long": 0,
        "max_short": 0,
        "max_abs": 0,
    }
    for sym in SYMBOLS
}

state_lock = Lock()  # for thread-safe writes to thread_state

# Track which orders we've already processed for executions
processed_order_ids = set()
processed_lock = Lock()

###############################################################################
# ------------------------ HELPER FUNCTIONS ----------------------------------
###############################################################################

def get_safe_mid(best, sym):
    """Midprice with SHIFT-safe fallbacks."""
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

    if last_valid_mid[sym] is not None:
        return last_valid_mid[sym]

    print(f"[WARN] No valid mid for {sym}, using 0.01")
    return 0.01


def estimate_sigma_garch(returns):
    """GARCH(1,1) on log returns with mild scaling and safe fallback."""
    returns = np.array(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    returns = returns[np.abs(returns) > 1e-12]

    if len(returns) < 10:
        return 0.50

    scaled = returns * 100.0

    try:
        am = arch_model(
            scaled,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",
            dist="normal",
            rescale=False,
        )
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=1)
        var_t = forecast.variance.iloc[-1, 0]

        sigma_scaled = float(np.sqrt(var_t))
        sigma_t = sigma_scaled / 100.0

        sigma_t = max(1e-4, min(sigma_t, 1.0))
        return sigma_t

    except Exception as e:
        print(f"[GARCH ERROR] {e}")
        if len(returns) > 1:
            sigma_t = returns.std()
            sigma_t = max(1e-4, min(float(sigma_t), 1.0))
            return sigma_t
        return 0.50


def compute_dynamic_size(q):
    """
    AS-style inventory-dependent size.
    q in shares; we cap size between 1 and 5 lots.
    """
    if q < 0:  # short → buy more aggressively
        phi_bid = phi_max
        phi_ask = phi_max * np.exp(-eta * q)
    else:      # long → sell more aggressively
        phi_bid = phi_max * np.exp(-eta * q)
        phi_ask = phi_max

    size_bid = min(max(1, int(phi_bid)), 5)
    size_ask = min(max(1, int(phi_ask)), 5)
    return size_bid, size_ask


def compute_as_quotes(mid, q, sigma, gamma, kappa, T_minus_t):
    """
    Pure Avellaneda–Stoikov quoting.

    r_t     = s_t − q_t γ σ² (T − t)
    δ_a+δ_b = γ σ² (T − t) + 2 ln(1 + γ/κ)
    bid     = r_t − δ, ask = r_t + δ
    """
    r = mid - q * gamma * (sigma ** 2) * T_minus_t
    spread = gamma * (sigma ** 2) * T_minus_t + 2.0 * np.log(1.0 + gamma / kappa)
    delta = spread / 2.0

    bid = round(r - delta, 2)
    ask = round(r + delta, 2)

    if bid >= ask:
        bid = ask - 0.01
    bid = max(0.01, bid)

    return bid, ask


def cancel_orders(trader, ticker):
    """Cancel all resting orders for this ticker."""
    for order in trader.get_waiting_list():
        if order.symbol == ticker:
            trader.submit_cancellation(order)
            sleep(0.1)


def close_positions(trader, ticker):
    """Close any long/short inventory for this ticker using market orders."""
    item = trader.get_portfolio_item(ticker)
    long_shares = item.get_long_shares()
    short_shares = item.get_short_shares()

    if long_shares > 0:
        print(f"[CLOSE] {ticker} selling long {long_shares}")
        trader.submit_order(
            shift.Order(shift.Order.Type.MARKET_SELL, ticker, long_shares // 100)
        )

    if short_shares > 0:
        print(f"[CLOSE] {ticker} buying back short {short_shares}")
        trader.submit_order(
            shift.Order(shift.Order.Type.MARKET_BUY, ticker, short_shares // 100)
        )

    sleep(0.5)


###############################################################################
# ------------------------ STRATEGY (PER THREAD) -----------------------------
###############################################################################

def strategy(trader: shift.Trader, ticker: str, endtime):
    """
    Threaded strategy for a single ticker.
    Runs AS + GARCH, logs executions, and updates global thread_state.
    """
    print(f"[START STRATEGY] {ticker}")

    # baseline P&L
    with state_lock:
        thread_state[ticker]["initial_pl"] = trader.get_portfolio_item(
            ticker
        ).get_realized_pl()

    while trader.get_last_trade_time() < endtime:
        # cancel previous quotes
        cancel_orders(trader, ticker)

        # get mid & inventory
        best_price = trader.get_best_price(ticker)
        mid = get_safe_mid(best_price, ticker)
        inv = trader.get_portfolio_item(ticker).get_shares()

        print(f"[{ticker}] Mid={mid:.4f} Inv={inv}")

        # update inventory extremes
        with state_lock:
            thread_state[ticker]["max_long"] = max(
                thread_state[ticker]["max_long"], inv
            )
            thread_state[ticker]["max_short"] = min(
                thread_state[ticker]["max_short"], inv
            )
            thread_state[ticker]["max_abs"] = max(
                thread_state[ticker]["max_abs"], abs(inv)
            )

        # update rolling mids
        rolling_mids[ticker].append(mid)
        mids_arr = np.array(rolling_mids[ticker], dtype=float)
        mids_arr = mids_arr[mids_arr > 0]

        rets = np.array([])
        if len(mids_arr) > 1:
            logp = np.log(mids_arr)
            rets = np.diff(logp)

        # GARCH sigma update
        if len(rets) >= 5:
            new_sigma = estimate_sigma_garch(rets)
            current_sigma[ticker] = new_sigma
            print(f"[{ticker}] σ updated → {new_sigma:.6f}")

        sigma = current_sigma[ticker]

        # compute AS quotes
        T_minus_t = 30.0  # local horizon in seconds (simple constant)
        bid_px, ask_px = compute_as_quotes(mid, inv, sigma, gamma, kappa, T_minus_t)
        size_bid, size_ask = compute_dynamic_size(inv)

        print(
            f"[{ticker}] Quotes: BID={bid_px}x{size_bid}  "
            f"ASK={ask_px}x{size_ask}  σ={sigma:.6f}"
        )

        # submit new quotes
        trader.submit_order(
            shift.Order(shift.Order.Type.LIMIT_BUY, ticker, size_bid, bid_px)
        )
        trader.submit_order(
            shift.Order(shift.Order.Type.LIMIT_SELL, ticker, size_ask, ask_px)
        )

        # process executions for THIS ticker
        with processed_lock:
            submitted_orders = list(trader.get_submitted_orders())

        for order in submitted_orders:
            # only care about this ticker and orders with some execution
            if order.symbol != ticker or order.executed_size <= 0:
                continue

            with processed_lock:
                if order.id in processed_order_ids:
                    continue
                processed_order_ids.add(order.id)

            execs = trader.get_executed_orders(order.id)
            for ex in execs:
                if ex.executed_size <= 0 or ex.symbol != ticker:
                    continue

                side = "BUY" if "BUY" in str(ex.type) else "SELL"
                qty = ex.executed_size * 100
                px = ex.executed_price

                print(f"[EXEC] {ticker}: {side} {qty} @ {px}")

                with state_lock:
                    thread_state[ticker]["trades"].append(
                        {"side": side, "qty": qty, "price": px}
                    )

        sleep(0.5)

    # after endtime: clean up
    cancel_orders(trader, ticker)
    close_positions(trader, ticker)

    print(f"[END STRATEGY] {ticker}")


###############################################################################
# ------------------------ FINAL REPORT --------------------------------------
###############################################################################

def final_report(trader: shift.Trader):
    print("\n================================================")
    print("              FINAL PERFORMANCE REPORT")
    print("================================================\n")

    total_net = 0.0

    for sym in SYMBOLS:
        print(f"\n----- {sym} -----")

        with state_lock:
            initial = thread_state[sym]["initial_pl"]
            trades = list(thread_state[sym]["trades"])
            max_long = thread_state[sym]["max_long"]
            max_short = thread_state[sym]["max_short"]
            max_abs = thread_state[sym]["max_abs"]

        final_pl = trader.get_portfolio_item(sym).get_realized_pl()
        net = final_pl - initial
        total_net += net

        print(f"Initial P&L:         {initial:.2f}")
        print(f"Final P&L:           {final_pl:.2f}")
        print(f"Net P&L:             {net:.2f}")
        print(f"Max Long Inventory:  {max_long}")
        print(f"Max Short Inventory: {max_short}")
        print(f"Max Abs Inventory:   {max_abs}")
        print(f"Total Executions:    {len(trades)}")

        if trades:
            buys = [t for t in trades if t["side"] == "BUY"]
            sells = [t for t in trades if t["side"] == "SELL"]

            def wavg(ts):
                return (
                    sum(t["qty"] * t["price"] for t in ts) / sum(t["qty"] for t in ts)
                    if ts
                    else 0.0
                )

            avg_buy = wavg(buys)
            avg_sell = wavg(sells)

            print(f"Avg Buy Price:       {avg_buy:.4f}")
            print(f"Avg Sell Price:      {avg_sell:.4f}")

    print("\n================================================")
    print(f"TOTAL STRATEGY NET P&L: {total_net:.2f}")
    print("================================================\n")


###############################################################################
# ------------------------ MAIN (SHIFT / LEADERBOARD) ------------------------
###############################################################################

def main(trader: shift.Trader):
    current_time = trader.get_last_trade_time()
    start_time = current_time
    end_time = start_time + timedelta(minutes=TRADE_DURATION_MINUTES)

    print(f"[CONFIG] Trading for {TRADE_DURATION_MINUTES} minutes")
    print(f"[TIME] start={start_time}  end={end_time}")

    threads = [
        Thread(target=strategy, args=(trader, sym, end_time))
        for sym in SYMBOLS
    ]

    # start threads
    for t in threads:
        t.start()
        sleep(0.3)

    # wait until end time
    while trader.get_last_trade_time() < end_time:
        sleep(1)

    # join all strategy threads
    for t in threads:
        t.join()

    # safety: ensure all positions closed
    for sym in SYMBOLS:
        cancel_orders(trader, sym)
        close_positions(trader, sym)

    final_report(trader)


###############################################################################
# ------------------------ ENTRYPOINT ----------------------------------------
###############################################################################

if __name__ == "__main__":
    with shift.Trader("") as trader:
        trader.connect("initiator.cfg", "")
        sleep(1)

        trader.sub_all_order_book()
        sleep(1)

        main(trader)
