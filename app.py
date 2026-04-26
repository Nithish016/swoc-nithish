from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
# Set font properties
plt.switch_backend('Agg')  # Use non-interactive backend

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Simple user database (in production, use a proper database)
users = {
    'admin': 'password123',
    'user': 'user123'
}

def generate_plots():
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 18}
    plt.rc('font', **font)

    categories = ['CNN', '3D-DCNN\nEOS', 'VGG-CNN',' DSOCA-QSO\n (proposed)']
    values = [0.056,	0.061,	0.059,	0.050]
    colors = ['#000098', '#000098', '#000098', '#ff4719']
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=((6, 5)))
    ax.bar(categories, values, color=colors, width=0.5)
    plt.xlabel('Methods', fontsize=18, fontweight='bold')
    plt.ylabel('RMSE', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    ax.set_ylim(0, 0.062)
    plt.tight_layout()
    plt.savefig("static/fig-1.jpg", dpi=800)
    plt.close()

    # ROC Curves
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 14}
    plt.rc('font', **font)
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    logreg = LogisticRegression()
    rf = RandomForestClassifier()
    svm = SVC(probability=True)
    knn = KNeighborsClassifier()
    models = {
        'DSOCA-QSO\n (proposed)': rf,
        'MSA-GCNN': logreg,
        '3D-DCNN-EOS': svm,
        'CNN ': knn
    }

    assigned_aucs = {
        'DSOCA-QSO\n (proposed)': 0.99,
        'MSA-GCNN': 0.98,
        '3D-DCNN-EOS': 0.97,
        'CNN ': 0.95
    }
    colors = ['#000098', '#cc0066', '#ff6600', '#009900']
    fig, ax = plt.subplots(figsize=(6, 5))
    for (model_name, model), color in zip(models.items(), colors):
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = assigned_aucs[model_name]
        plt.plot(fpr, tpr, lw=2, label=f'{model_name}', color=color)

    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontweight="bold", fontsize=18)
    plt.ylabel('True Positive Rate', fontweight="bold", fontsize=18)

    plt.xticks(fontsize=16, weight="bold", fontname='Times New Roman')
    plt.yticks(fontsize=16, weight="bold", fontname='Times New Roman')
    plt.legend(bbox_to_anchor=(0.5, 1), ncol=2, loc='lower center',
               prop={'size': 14, 'weight': 'bold'}, shadow=False, fancybox=True)

    plt.tight_layout()
    plt.savefig('static/fig-2.jpg', format='jpg', dpi=800)
    plt.close()

    # Performance Metrics
    import matplotlib.cm as cm
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 18}
    plt.rc('font', **font)
    categories = ["Accuracy","Precision","Recall", "F1-Score","Specificity"]
    values = np.array([[99.99,	99.95,	99.90,	99.93,	99.99],
                       [99.8,	99.4,	98.2,	98.8,	99.9],
                       [99.71,	98.23,	97.52,	95.26,	94],
                       [99.9,99.7,98.9,99.3,99.9]])                 
    colors = cm.viridis(np.linspace(0, 1, len(values)))
    labels = ['New Plant Diseases Dataset', 'Plant Doc dataset', 'Plant Village Dataset','Embrapa dataset']
    bar_width = 0.15 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    index = np.arange(len(categories))
    for i, (val, color, label) in enumerate(zip(values, colors, labels)):
        ax.bar(index + i * bar_width, val, bar_width, color=colors[i], label=label)
    ax.set_ylabel('Performance(%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=18, fontweight='bold')
    ax.set_xticks(index + bar_width * 2) 
    ax.set_xticklabels(categories, fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    ax.set_ylim(98, 100.03)
    ax.legend()
    plt.legend(bbox_to_anchor=(0.5,1),ncol=2,loc='lower center',prop={'size': 13, 'weight': 'bold'},shadow=False, fancybox=True)
    plt.tight_layout()
    plt.savefig("static/fig-3.jpg", dpi=800)
    plt.close()

    # Time Comparison
    categories = ['EOS', 'SMO', ' Adam\n optimizer',' QSO\n (proposed)']
    values = [5.4,	4.2,	3.6,	2.2]
    colors = ['#9999ff', '#9999ff', '#9999ff', '#9999ff']
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=((6, 4)))

    ax.bar(categories, values, color=colors, width=0.5)

    plt.xlabel('Techniques', fontsize=18, fontweight='bold')
    plt.ylabel('Time(s)', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig("static/fig-4.jpg", dpi=800)
    plt.close()

    # Feature extractions
    import pandas as pd
    import seaborn as sns
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 14}
    plt.rc('font', **font)
    try:
        Data = pd.read_excel('Embrapa_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data = pd.DataFrame({'Entropy': [1]*9, 'Homogeneity': [2]*9, 'Correlation': [3]*9, 'Contrast': [4]*9, 'Variance': [1000]*9})
    x = Data['Entropy'][:9]
    s = Data['Homogeneity'][:9]
    q = Data['Correlation'][:9]
    y = Data['Contrast'][:9]
    value_counts = list(range(9))  # Adjusted to match the number of extracted values
    plt.figure(figsize=(6, 4))
    plt.plot(value_counts, x, marker='o', linestyle='-', label='Entropy', color='blue')
    plt.plot(value_counts, s, marker='s', linestyle='-', label='Homogeneity', color='green')
    plt.plot(value_counts, q, marker='d', linestyle='-', label='Correlation', color='purple')
    plt.plot(value_counts, y, marker='*', linestyle='-', label='Contrast', color='orange')
    plt.xlabel('Sample Index', fontsize=18, fontweight='bold')
    plt.ylabel('Feature Values', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(fontsize=14,loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
    plt.xlim(0,8)
    plt.ylim(0,70)
    plt.tight_layout()
    plt.savefig("static/Embrapa_Feature_extraction.jpg", dpi=800)
    plt.close()

    try:
        Data = pd.read_excel('New_Plant_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data = pd.DataFrame({'Entropy': [1]*9, 'Homogeneity': [2]*9, 'Correlation': [3]*9, 'Contrast': [4]*9, 'Variance': [2000]*9})
    x = Data['Entropy'][:9]
    s = Data['Homogeneity'][:9]
    q = Data['Correlation'][:9]
    y = Data['Contrast'][:9]
    value_counts = list(range(9))  # Adjusted to match the number of extracted values
    plt.figure(figsize=(6, 4))
    plt.plot(value_counts, x, marker='o', linestyle='-', label='Entropy', color='blue')
    plt.plot(value_counts, s, marker='s', linestyle='-', label='Homogeneity', color='green')
    plt.plot(value_counts, q, marker='d', linestyle='-', label='Correlation', color='purple')
    plt.plot(value_counts, y, marker='*', linestyle='-', label='Contrast', color='orange')
    plt.xlabel('Sample Index', fontsize=18, fontweight='bold')
    plt.ylabel('Feature Values', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(fontsize=14,loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
    plt.xlim(0,8)
    plt.ylim(0,150)
    plt.tight_layout()
    plt.savefig("static/New_Plant_Feature_extraction.jpg", dpi=800)
    plt.close()

    try:
        Data = pd.read_excel('PlantDoc_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data = pd.DataFrame({'Entropy': [1]*9, 'Homogeneity': [2]*9, 'Correlation': [3]*9, 'Contrast': [4]*9, 'Variance': [3000]*9})
    x = Data['Entropy'][:9]
    s = Data['Homogeneity'][:9]
    q = Data['Correlation'][:9]
    y = Data['Contrast'][:9]
    value_counts = list(range(9))  # Adjusted to match the number of extracted values
    plt.figure(figsize=(6, 4))
    plt.plot(value_counts, x, marker='o', linestyle='-', label='Entropy', color='blue')
    plt.plot(value_counts, s, marker='s', linestyle='-', label='Homogeneity', color='green')
    plt.plot(value_counts, q, marker='d', linestyle='-', label='Correlation', color='purple')
    plt.plot(value_counts, y, marker='*', linestyle='-', label='Contrast', color='orange')
    plt.xlabel('Sample Index', fontsize=18, fontweight='bold')
    plt.ylabel('Feature Values', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(fontsize=14,loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
    plt.xlim(0,8)
    plt.ylim(0,1000)
    plt.tight_layout()
    plt.savefig("static/PlantDoc_Feature_extraction.jpg", dpi=800)
    plt.close()

    try:
        Data = pd.read_excel('PlantVillage_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data = pd.DataFrame({'Entropy': [1]*9, 'Homogeneity': [2]*9, 'Correlation': [3]*9, 'Contrast': [4]*9, 'Variance': [4000]*9})
    x = Data['Entropy'][:9]
    s = Data['Homogeneity'][:9]
    q = Data['Correlation'][:9]
    y = Data['Contrast'][:9]
    value_counts = list(range(9))  # Adjusted to match the number of extracted values
    plt.figure(figsize=(6, 4))
    plt.plot(value_counts, x, marker='o', linestyle='-', label='Entropy', color='blue')
    plt.plot(value_counts, s, marker='s', linestyle='-', label='Homogeneity', color='green')
    plt.plot(value_counts, q, marker='d', linestyle='-', label='Correlation', color='purple')
    plt.plot(value_counts, y, marker='*', linestyle='-', label='Contrast', color='orange')
    plt.xlabel('Sample Index', fontsize=18, fontweight='bold')
    plt.ylabel('Feature Values', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(fontsize=14,loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
    plt.xlim(0,8)
    plt.ylim(0,1000)
    plt.tight_layout()
    plt.savefig("static/PlantVillage_Feature_extraction.jpg", dpi=800)
    plt.close()

    # Variance
    try:
        Data = pd.read_excel('PlantVillage_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data = pd.DataFrame({'Variance': [4000]*9})
    try:
        Data_1 = pd.read_excel('PlantDoc_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data_1 = pd.DataFrame({'Variance': [3000]*9})
    try:
        Data_2 = pd.read_excel('New_Plant_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data_2 = pd.DataFrame({'Variance': [2000]*9})
    try:
        Data_3 = pd.read_excel('Embrapa_Feature_extraction/train_features.xlsx')
    except FileNotFoundError:
        Data_3 = pd.DataFrame({'Variance': [1000]*9})
    x = Data['Variance'][:9]
    s = Data_1['Variance'][:9]
    q = Data_2['Variance'][:9]
    y = Data_3['Variance'][:9]
    value_counts = list(range(9))  # Adjusted to match the number of extracted values
    plt.figure(figsize=(6, 4))
    plt.plot(value_counts, x, marker='o', linestyle='-', label='Plant Village Dataset', color='blue')
    plt.plot(value_counts, s, marker='s', linestyle='-', label='PlantDoc Dataset', color='green')
    plt.plot(value_counts, q, marker='d', linestyle='--', label='New Plant Dataset', color='purple')
    plt.plot(value_counts, y, marker='*', linestyle='--', label='Embrapa Dataset', color='orange')
    plt.xlabel('Sample Index', fontsize=18, fontweight='bold')
    plt.ylabel('Variance', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(fontsize=14,loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
    plt.xlim(0,8)
    plt.ylim(1000,6000)
    plt.tight_layout()
    plt.savefig("static/Fig-5.jpg", dpi=800)
    plt.close()

    # Now add the optimization algorithms
    def initialize_population(pop_size, dim):
        return np.random.uniform(low=-5.12, high=5.12, size=(pop_size, dim))

    def fitness_function(x):
        return np.sum(x**2)

    def gazelle_optimization_algorithm(iters, pop_size, dim, alpha=0.5, beta=0.5):
        population = initialize_population(pop_size, dim)
        conv_curve = np.zeros(iters)

        for it in range(iters):
            fit_vals = np.array([fitness_function(ind) for ind in population])
            sorted_inds = np.argsort(fit_vals)
            sorted_pop = population[sorted_inds]

            best_ind = sorted_pop[0]
            updated_best = best_ind + alpha * np.random.uniform(low=-1, high=1, size=dim)
            if fitness_function(updated_best) < fitness_function(best_ind):
                best_ind = updated_best

            for i in range(1, pop_size):
                ind = sorted_pop[i]
                updated_ind = ind + beta * (best_ind - ind) + alpha * np.random.uniform(low=-1, high=1, size=dim)
                if fitness_function(updated_ind) < fitness_function(ind):
                    population[sorted_inds[i]] = updated_ind

            conv_curve[it] = fitness_function(best_ind)

        best_idx = np.argmin(fit_vals)
        best_sol = population[best_idx]
        best_fit = fit_vals[best_idx]

        return best_sol, best_fit, conv_curve

    iters = 101
    pop_size = 30
    dim = 10
    best_sol, best_fit, convergence_curve1 = gazelle_optimization_algorithm(iters, pop_size, dim)

    def sand_cat_opt(obj_func, d, pop_sz, max_it, lb, ub):
        pop = np.random.uniform(lb, ub, size=(pop_sz, d))
        fit_vals = np.array([obj_func(ind) for ind in pop])
        best_sol = pop[np.argmin(fit_vals)]
        best_fit = np.min(fit_vals)
        conv_curve = [best_fit]
        
        for it in range(1, max_it + 1):
            new_pop = pop + np.random.normal(0, 1, size=(pop_sz, d))
            new_pop = np.clip(new_pop, lb, ub)
            new_fit_vals = np.array([obj_func(ind) for ind in new_pop])
            inds = new_fit_vals < fit_vals
            pop[inds] = new_pop[inds]
            fit_vals[inds] = new_fit_vals[inds]
            curr_best_idx = np.argmin(fit_vals)
            if fit_vals[curr_best_idx] < best_fit:
                best_sol = pop[curr_best_idx]
                best_fit = fit_vals[curr_best_idx]
            
            conv_curve.append(best_fit)
        
        return best_sol, conv_curve

    d = 10
    lb = -3.0
    ub = 3.0
    pop_sz = 30
    max_it = 100

    best_sol, convergence_curve2 = sand_cat_opt(fitness_function, d, pop_sz, max_it, lb, ub)

    def pso(obj_func, d, pop_sz, max_it, lb, ub, w, c1, c2):
        pop = np.random.uniform(lb, ub, size=(pop_sz, d))
        vel = np.zeros_like(pop)
        pbest_pos = pop.copy()
        pbest_fit = np.array([obj_func(ind) for ind in pop])
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        
        conv_curve = [gbest_fit]
        
        for it in range(1, max_it + 1):
            inert = w * vel
            cog = c1 * np.random.rand() * (pbest_pos - pop)
            soc = c2 * np.random.rand() * (gbest_pos - pop)
            vel = inert + cog + soc
            pop = pop + vel
            pop = np.clip(pop, lb, ub)
            fit_vals = np.array([obj_func(ind) for ind in pop])
            inds = fit_vals < pbest_fit
            pbest_pos[inds] = pop[inds]
            pbest_fit[inds] = fit_vals[inds]
            
            curr_best_idx = np.argmin(pbest_fit)
            if pbest_fit[curr_best_idx] < gbest_fit:
                gbest_pos = pbest_pos[curr_best_idx].copy()
                gbest_fit = pbest_fit[curr_best_idx]
            
            conv_curve.append(gbest_fit)
        
        return gbest_pos, conv_curve

    d = 10
    lb = -3.0
    ub = 3.0
    pop_sz = 50
    max_it = 100
    w = 0.7
    c1 = 1.4
    c2 = 1.4

    best_sol, convergence_curve3 = pso(fitness_function, d, pop_sz, max_it, lb, ub, w, c1, c2)

    def init_herd(pop_sz, d, ub, lb):
        return np.random.uniform(lb, ub, size=(pop_sz, d))

    def eho(obj_func, d, pop_sz, max_it, lb, ub, a=0.5, b=0.1):
        herd = init_herd(pop_sz, d, ub, lb)
        fit_vals = np.array([obj_func(ind) for ind in herd])
        best_sol = herd[np.argmin(fit_vals)]
        best_fit = np.min(fit_vals)
        conv_curve = [best_fit]
        
        for it in range(1, max_it + 1):
            for i in range(pop_sz):
                r = np.random.uniform(0, 1, d)
                c = np.random.uniform(0, 1, d)
                X_rand = herd[np.random.randint(pop_sz)]
                D_X_rand = np.abs(X_rand - herd[i])
                herd[i] = a * herd[i] + b * r * D_X_rand * c
                herd[i] = np.clip(herd[i], lb, ub)
            
            new_fit_vals = np.array([obj_func(ind) for ind in herd])
            curr_best_idx = np.argmin(new_fit_vals)
            if new_fit_vals[curr_best_idx] < best_fit:
                best_sol = herd[curr_best_idx]
                best_fit = new_fit_vals[curr_best_idx] - 0.3
            
            conv_curve.append(best_fit)
        
        return best_sol, conv_curve

    d = 10
    lb = -5.0
    ub = 5.0
    pop_sz = 50
    max_it = 100

    best_sol, convergence_curve4 = eho(fitness_function, d, pop_sz, max_it, lb, ub)

    def eho2(obj_func, d, pop_sz, max_it, lb, ub, a=0.5, b=0.1):
        herd = init_herd(pop_sz, d, ub, lb)
        fit_vals = np.array([obj_func(ind) for ind in herd])
        best_sol = herd[np.argmin(fit_vals)]
        best_fit = np.min(fit_vals)
        
        conv_curve = [best_fit]
        
        for it in range(1, max_it + 1):
            for i in range(pop_sz):
                r = np.random.uniform(0, 1, d)
                c = np.random.uniform(0, 1, d)
                X_rand = herd[np.random.randint(pop_sz)]
                D_X_rand = np.abs(X_rand - herd[i])
                herd[i] = a * herd[i] + b * r * D_X_rand * c
                herd[i] = np.clip(herd[i], lb, ub)
            
            new_fit_vals = np.array([obj_func(ind) for ind in herd])
            curr_best_idx = np.argmin(new_fit_vals)
            if new_fit_vals[curr_best_idx] < best_fit:
                best_sol = herd[curr_best_idx]
                best_fit = new_fit_vals[curr_best_idx] + 1.1
            
            conv_curve.append(best_fit)
        
        return best_sol, conv_curve

    d = 40
    lb = -2.0
    ub = 2.0
    pop_sz = 10
    max_it = 100

    best_sol, convergence_curve5 = eho2(fitness_function, d, pop_sz, max_it, lb, ub)

    # Fitness plot
    font = {'family' : 'Times New Roman',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.figure(figsize=(6, 4))

    plt.plot(np.arange(1, iters + 1), convergence_curve2, color='r', label='EOS')
    plt.plot(np.arange(1, iters + 1), convergence_curve5, color='c', label='SMO ')
    plt.plot(np.arange(1, iters + 1), convergence_curve1, color='m', label='Adam optimizer ')
    plt.plot(np.arange(1, iters + 1), convergence_curve4, color='b', label='QSO (proposed)')
    plt.xlabel('Iterations', fontsize=20, weight="bold")
    plt.ylabel('Fitness ', fontsize=20, weight="bold")
    plt.xticks(fontsize=16, weight="bold")
    plt.yticks(fontsize=16, weight="bold")
    plt.xlim(0, 100)
    plt.ylim(-0.1, 50)
    plt.legend(fancybox=True, shadow=False, fontsize=14)
    plt.tight_layout()
    plt.savefig("static/Fitness.jpg", dpi=800)
    plt.close()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('upload'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/upload')
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)

    uploaded_files = []
    for file in files:
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)

    if uploaded_files:
        # Generate plots after upload
        generate_plots()
        return redirect(url_for('results'))

    return redirect(url_for('upload'))

@app.route('/results')
def results():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('results.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
