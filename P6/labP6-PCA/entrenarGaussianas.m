function modelo = entrenarGaussianas( Xtr, ytr, NaiveBayes, landa, n1, n2 )
    % Entrena una Gaussana para cada clase y devuelve:
    % modelo{i}.N     : Numero de muestras de la clase i
    % modelo{i}.mu    : Media de la clase i
    % modelo{i}.Sigma : Covarianza de la clase i
    % Si NaiveBayes = 1, las matrices de Covarianza serán diagonales porque no 
    %                    tiene en cuenta las covarianzas
    % Se regularizarán las covarianzas mediante: Sigma = Sigma + landa*eye(D)

    %Valores con los que se va a clasificar
    valores = [n1 , n2];

    % Bucle con los dos valores
    for i = valores
        buenos = (ytr==i);
        modelo{i}.N = size(find(ytr==i),1);
        modelo{i}.Sigma = cov(Xtr(buenos,:));
        modelo{i}.mu = mean(Xtr(buenos,:));
        modelo{i}.mu = modelo{i}.mu';
        modelo{i}.Sigma = modelo{i}.Sigma + landa*eye(size(modelo{i}.Sigma));
        if(NaiveBayes==1)
            modelo{i}.Sigma = diag(diag(modelo{i}.Sigma));
        end 
    end
end