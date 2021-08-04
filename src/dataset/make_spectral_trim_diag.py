import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


lower_end = 0.4 * 1000
upper_end = 2.4 * 1000
params = {
  'min_width': 0.35 * 1000, 'max_lower': 0.7 * 1000, 'min_upper': 1.0 * 1000,
  'lower_std': 0.1 * 1000, 'upper_std': 0.5 * 1000
}

lower_dom = np.linspace(lower_end, params['max_lower'], 1000)
lower_pdf = norm.pdf(lower_dom, loc=lower_end, scale=params['lower_std'])
upper_dom = np.linspace(params['min_upper'], upper_end, 1000)
upper_pdf = norm.pdf(upper_dom, loc=upper_end, scale=params['upper_std'])

plt.figure(figsize=(6, 3.5))
plt.plot(lower_dom, lower_pdf, label='Lower end PDF')
plt.fill_between(lower_dom, lower_pdf, alpha=0.3)
plt.plot(upper_dom, upper_pdf, label='Upper end PDF')
plt.fill_between(upper_dom, upper_pdf, alpha=0.3)
plt.legend(loc='upper right')
plt.xticks(np.arange(lower_end, upper_end+1, 300).tolist() + [upper_end])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Probability density')
plt.tight_layout()
plt.show()
