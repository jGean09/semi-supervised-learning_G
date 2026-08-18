import numpy as np
from sklearn.base import clone
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_array


class MySelfNewEssembleCP:
    def __init__(
        self,
        base_estimator,
        criterion="threshold",
        threshold=0.75,
        k_best=10,
        max_iter=10,
        silhouette_threshold=-0.2,
        verbose=True
    ):
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.silhouette_threshold = silhouette_threshold
        self.classifier_ = None
        self.transduction_ = None
        self.n_iter_ = 0
        self.termination_condition_ = None

    def fit(self, X, y):
        X = check_array(X)
        self.classifier_ = clone(self.base_estimator)

        self.transduction_ = np.copy(y)
        labeled_mask = y != -1

        # Treinamento inicial
        self.classifier_.fit(X[labeled_mask], y[labeled_mask])

        while not np.all(labeled_mask) and self.n_iter_ < self.max_iter:
            self.n_iter_ += 1

            # pega o vetor de rotulados e inverte para ter os não-rotulados
            unlabeled_mask = ~labeled_mask
            X_unlabeled = X[unlabeled_mask]

            if X_unlabeled.shape[0] == 0:
                break

            probs = self.classifier_.predict_proba(X_unlabeled)
            max_probs = probs.max(axis=1)

            if self.criterion == "threshold":
                confident_samples = max_probs >= self.threshold
            elif self.criterion == "k_best":
                n_to_select = min(self.k_best, max_probs.shape[0])
                confident_samples = np.zeros_like(max_probs, dtype=bool)
                if n_to_select > 0:
                    confident_indices = np.argpartition(-max_probs, n_to_select)[:n_to_select]
                    confident_samples[confident_indices] = True
            else:
                raise ValueError("Invalid criterion.")

            confident_indices = np.array([], dtype=int)

            if not np.any(confident_samples):
                # Todas as instâncias não rotuladas que eu classifiquei não
                # tiveram confiança acima do threshold,
                # ou seja vetor [False] * n ?
                # Então vamos calcular o índice Silhouette para cada instância
                # e avaliar se as instâncias podem pertencer ao grupo
                # (valor do índice < threshold do índice).
                # Silhouette filtering e colocar essas instâncias fracas no 
                # conjunto dos não rotulados para assim, reclassificar as
                # instâncias fracas.
                X_labeled = X[labeled_mask]
                y_labeled = self.transduction_[labeled_mask]
                silhouette_vals = silhouette_samples(X_labeled, y_labeled)
                weak_mask = silhouette_vals < self.silhouette_threshold

                if np.any(weak_mask):
                    # Se tiver alguma instância com o índice menor que o
                    # threshold. Então, remove os pseudo-rótulos problemáticos
                    # e retorna as instância ao conjunto dos não rotulados.

                    ##------------------------------------------------------------------------------
                    ## Remove os pseudo-rótulos das instâncias consideradas fracas pelo Silhouette
                    ## e as retorna ao conjunto de dados não rotulados.
                    indices_labeled = np.where(labeled_mask)[0] 
                    indices_weak = indices_labeled[weak_mask]   
                    labeled_mask[indices_weak] = False          
                    self.transduction_[indices_weak] = -1    
                    ##------------------------------------------------------------------------------

                    if self.verbose:
                        print(f"{len(indices_weak)} pseudo-rótulos removidos pelo silhouette e as instâncias vão para o conjuntos dos não rotulados.")

                else:
                    self.termination_condition_ = "no_change"
                    if self.verbose:
                        print("Nenhuma instância confiável ou fraca encontrada. Encerrando.")
                    break


            # Atualiza treinamento com todos os rótulos atuais
            X_labeled = X[labeled_mask]
            y_labeled = self.transduction_[labeled_mask]
            self.classifier_.fit(X_labeled, y_labeled)   ## Re-treina o especialista com o conjunto rotulado atualizado.

            if self.verbose:
                print(f"Iteração {self.n_iter_}: {np.sum(confident_samples)} novas amostras confiantes adicionadas.")

            indices_unlabeled = np.where(unlabeled_mask)[0] 
            confident_indices = indices_unlabeled[confident_samples] 

            ## Atualiza os rótulos das instâncias confiantes e as move para o conjunto rotulado.
            if confident_indices.size > 0:
                new_y = self.classifier_.predict(X[confident_indices])
                self.transduction_[confident_indices] = new_y
                labeled_mask[confident_indices] = True

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        elif np.all(labeled_mask):
            self.termination_condition_ = "all_labeled"

    def predict(self, X):
        if self.classifier_ is None:
            raise ValueError("O classificador ainda não foi treinado.")
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        if self.classifier_ is None:
            raise ValueError("O classificador ainda não foi treinado.")
        return self.classifier_.predict_proba(X)


class MySelfNewEssembleCP:
    def __init__(
        self,
        base_estimator,
        criterion="threshold",
        threshold=0.75,
        k_best=10,
        max_iter=10,
        silhouette_threshold=-0.2,
        verbose=False
    ):
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.silhouette_threshold = silhouette_threshold
        self.classifier_ = None
        self.transduction_ = None
        self.n_iter_ = 0
        self.termination_condition_ = None

    def fit(self, X, y):
        X = check_array(X)
        self.classifier_ = clone(self.base_estimator)

        self.transduction_ = np.copy(y)
        labeled_mask = y != -1

        # Treinamento inicial
        self.classifier_.fit(X[labeled_mask], y[labeled_mask])

        while not np.all(labeled_mask) and self.n_iter_ < self.max_iter:
            self.n_iter_ += 1

            # pega o vetor de rotulados e inverte para ter os não-rotulados
            unlabeled_mask = ~labeled_mask
            X_unlabeled = X[unlabeled_mask]

            if X_unlabeled.shape[0] == 0:
                break

            probs = self.classifier_.predict_proba(X_unlabeled)
            max_probs = probs.max(axis=1)

            if self.criterion == "threshold":
                confident_samples = max_probs >= self.threshold
            elif self.criterion == "k_best":
                n_to_select = min(self.k_best, max_probs.shape[0])
                confident_samples = np.zeros_like(max_probs, dtype=bool)
                if n_to_select > 0:
                    confident_indices = np.argpartition(-max_probs, n_to_select)[:n_to_select]
                    confident_samples[confident_indices] = True
            else:
                raise ValueError("Invalid criterion.")

            confident_indices = np.array([], dtype=int)

            if not np.any(confident_samples):
                # Todas as instâncias não rotuladas que eu classifiquei não
                # tiveram confiança acima do threshold,
                # ou seja vetor [False] * n ?
                # Então vamos calcular o índice Silhouette para cada instância
                # e avaliar se as instâncias podem pertencer ao grupo
                # (valor do índice < threshold do índice).
                # Silhouette filtering e colocar essas instâncias fracas no 
                # conjunto dos não rotulados para assim, reclassificar as
                # instâncias fracas.
                X_labeled = X[labeled_mask]
                y_labeled = self.transduction_[labeled_mask]
                silhouette_vals = silhouette_samples(X_labeled, y_labeled)
                weak_mask = silhouette_vals < self.silhouette_threshold

                if np.any(weak_mask):
                    # Se tiver alguma instância com o índice menor que o
                    # threshold. Então, remove os pseudo-rótulos problemáticos
                    # e retorna as instância ao conjunto dos não rotulados.

                    ##------------------------------------------------------------------------------
                    ## Remove os pseudo-rótulos das instâncias consideradas fracas pelo Silhouette
                    ## e as retorna ao conjunto de dados não rotulados.
                    indices_labeled = np.where(labeled_mask)[0] 
                    indices_weak = indices_labeled[weak_mask]   
                    labeled_mask[indices_weak] = False          
                    self.transduction_[indices_weak] = -1    
                    ##------------------------------------------------------------------------------

                    if self.verbose:
                        print(f"{len(indices_weak)} pseudo-rótulos removidos pelo silhouette e as instâncias vão para o conjuntos dos não rotulados.")

                else:
                    self.termination_condition_ = "no_change"
                    if self.verbose:
                        print("Nenhuma instância confiável ou fraca encontrada. Encerrando.")
                    break


            # Atualiza treinamento com todos os rótulos atuais
            X_labeled = X[labeled_mask]
            y_labeled = self.transduction_[labeled_mask]
            self.classifier_.fit(X_labeled, y_labeled)   ## Re-treina o especialista com o conjunto rotulado atualizado.

            if self.verbose:
                print(f"Iteração {self.n_iter_}: {np.sum(confident_samples)} novas amostras confiantes adicionadas.")

            indices_unlabeled = np.where(unlabeled_mask)[0] 
            confident_indices = indices_unlabeled[confident_samples] 

            ## Atualiza os rótulos das instâncias confiantes e as move para o conjunto rotulado.
            if confident_indices.size > 0:
                new_y = self.classifier_.predict(X[confident_indices])
                self.transduction_[confident_indices] = new_y
                labeled_mask[confident_indices] = True

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        elif np.all(labeled_mask):
            self.termination_condition_ = "all_labeled"

    def predict(self, X):
        if self.classifier_ is None:
            raise ValueError("O classificador ainda não foi treinado.")
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        if self.classifier_ is None:
            raise ValueError("O classificador ainda não foi treinado.")
        return self.classifier_.predict_proba(X)