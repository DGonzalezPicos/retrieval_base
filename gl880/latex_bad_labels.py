import matplotlib.pyplot as plt
from matplotlib import rc

# Use LaTeX for text rendering
rc('text', usetex=True)

# Corrected list of labels to test
labels = [
    '$\\log\\ \\mathrm{Ti}$', '$\\log\\ \\mathrm{FeH}$', '$\\log\\ \\mathrm{CN}$',
    '$\\log\\ \\mathrm{VO}$', '$\\log\\ \\mathrm{TiO}$', '$\\log\\ \\mathrm{H_2O}_0$',
    '$\\log\\ \\mathrm{H_2O}_1$', '$\\log\\ \\mathrm{H_2O}_2$', '$\\log\\ P_\\mathrm{H_2O}$',
    '$\\log\\ \\mathrm{OH}_0$', '$\\log\\ \\mathrm{OH}_1$', '$\\log\\ \\mathrm{OH}_2$',
    '$\\log\\ P_\\mathrm{OH}$'
]

# Function to test rendering of each label
def test_labels(labels):
    bad_labels = []
    for label in labels:
        try:
            plt.figure()
            plt.text(0.5, 0.5, label, fontsize=12)
            plt.title(f'Test: {label}')
            plt.show()
            plt.close()
            print(f"Label {label} rendered successfully")
        except Exception as e:
            print(f"Error with label: {label}\n{e}")
            bad_labels.append((label, str(e)))
    return bad_labels

# Test each label and capture bad labels
bad_labels = test_labels(labels)
for label, error in bad_labels:
    print(f"Error with label: {label}\n{error}")
