import matplotlib.pyplot as plt

expected_age = 90

current_age = 29

monthly_payment = 200

yearly_capital = 12 * monthly_payment

yearly_earning_factor = 1.06

yearly_rv_cost_factor = 0.975

yearly_fond_cost_factor = 0.99405

years_of_cost = 38

years_of_payment = expected_age - (current_age + years_of_cost)

current_amount_rv = 0

current_amount_fond = 0

free_of_tax = 801

capital_gain_tax = 0.25

income_tax_side_gains = 0.1635

fix_cost = 194

five_year_cost = 270

x_rv = []
x_fond = []
payouts_rv = []
payouts_fond = []
y = list(range(current_age, expected_age))

for year in range(years_of_cost):
    current_amount_rv += yearly_capital
    current_amount_rv = current_amount_rv * yearly_earning_factor * yearly_rv_cost_factor
    x_rv.append(current_amount_rv)

    current_amount_fond_prev = current_amount_fond + yearly_capital
    current_amount_fond = current_amount_fond_prev * yearly_earning_factor * yearly_fond_cost_factor
    # current_amount_fond = current_amount_fond - max(0.0, (current_amount_fond - current_amount_fond_prev - 801) * (1 - capital_gain_tax))
    x_fond.append(current_amount_fond)

    payouts_fond.append(0)
    payouts_rv.append(0)

for year in range(years_of_payment):
    payout_rv = (current_amount_rv / 10000) * 51.41 * 12
    current_amount_rv_prev = current_amount_rv - payout_rv
    current_amount_rv = current_amount_rv_prev * yearly_earning_factor
    current_amount_rv = current_amount_rv - max(0.0, (current_amount_rv - current_amount_rv_prev - 801) * (1 - income_tax_side_gains))
    x_rv.append(current_amount_rv)
    payouts_rv.append(payout_rv)

    payout_fond = (current_amount_fond / 10000) * 51.41 * 12
    current_amount_fond_prev = current_amount_fond - payout_fond
    current_amount_fond = current_amount_fond_prev * yearly_earning_factor * yearly_fond_cost_factor
    current_amount_fond = current_amount_fond - max(0.0, (current_amount_fond - current_amount_fond_prev - 801) * (1 - capital_gain_tax))
    x_fond.append(current_amount_fond)
    payouts_fond.append(payout_fond)

plt.plot(y, x_rv, color='red', label='RV')
plt.plot(y, x_fond, color='blue', label='Fond')
plt.plot(y, payouts_rv, color='green', label='RV Payout')
plt.plot(y, payouts_fond, color='brown', label='Fond Payout')

plt.xlabel('Age')
plt.ylabel('Euro')
plt.title('Yields of RV vs Fonds')
plt.legend(loc="best")

plt.show()

