
# ANALIZA PERFORMANSI KLASIFIKACIONIH ALGORITAMA

library(foreign)
library(ltm)
library(mice)
library(mda)# fda regresiju
library(randomForest)
library(caret)#train
library(RWeka)
library(psych)
library(lares)
library(xgboost)
library(tibble)
library(Amelia)
library(rpart)
library(factoextra)
library(FactoMineR)
library(factoextra)
library(devtools)
library(corrplot)
library(ROCR)
library(pROC)
library(magrittr)
library(dplyr)# %>%
library(Matrix)
library(class)
library(tidyr)
library(leaps)
library(glmnet)
library(e1071) # svm
library(rpart.plot)
library(psych)
library(stats)
library(MASS)
library(AICcmodavg)
library(car)
library(ROSE)

tek_seting <- getOption("warn")

options(warn = -1)


#https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

# 1. FORMIRANJE  DATA SETA i transformacija


# Import podataka

default_cr<-read.csv(choose.files())

# Provera balansiranosti podataka

round (prop.table( as.table(table(default_cr$default.payment.next.month))),2)


default_cr<-default_cr[,-1]# Uklanjam ID

set.seed(10)

default_cr<-default_cr[sample(1:nrow(default_cr)),]# 
head(default_cr)

# Provera zastupljenosti klasa kategorickih promenjivih,za default i no default klijente

xtabs(~SEX+default.payment.next.month,data=default_cr)
xtabs(~MARRIAGE+default.payment.next.month,data=default_cr)
xtabs(~EDUCATION+default.payment.next.month,data=default_cr)

# Provera prisustva  NAs 

broj_NAs<-sum(is.na(default_cr))

if (broj_NAs != 0) {
  
  cat("Ukupan broj NAs =",broj_NAs,"\n")
  
} else {
  
  print("Data set je kompletan")
}

# Uklanjanje uzoraka sa nepoznatom vrednost (0) za marriage i education

levels(as.factor(default_cr$MARRIAGE))
default_cr<-default_cr[default_cr$MARRIAGE!=0,]


levels(as.factor(default_cr$EDUCATION))
default_cr<-default_cr[default_cr$EDUCATION!=0,] # (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

# kreiranje  factor promenjive

default_cr$default.payment.next.month <- as.factor(default_cr$default.payment.next.month)
levels(default_cr$default.payment.next.month)
levels(default_cr$default.payment.next.month) <- c("Not Default", "Default")# 1 da, 0 ne 

norm_data<-default_cr# norm_data, podaci sa jednom katego. promenjivom, 

default_cr$MARRIAGE <- as.factor(default_cr$MARRIAGE)
levels(default_cr$MARRIAGE) <- c("Married" , "Single" ,"Others") # 1=married, 2=single, 3=others

default_cr$EDUCATION <- as.factor(default_cr$EDUCATION)
levels(default_cr$EDUCATION) <- c( "Graduate School", "University", "High school", "Others", "Unknown1", "Unknown2") 

default_cr$SEX <- as.factor(default_cr$SEX)
levels(default_cr$SEX) <- c("Male", "Female")#Gender (1=male, 2=female)


# 2. OPIS PODATAKA

# Descriptivna statistika data seta.
# Mean, Median, Q1, Q3, Min, Max, prisutvo NA's

str(default_cr)
summary(default_cr)
sapply(select_if(default_cr,is.numeric), skew)#  skewness izmedju -0.5 and 0.5, ditribucija je priblizno simetricna
sapply(select_if(default_cr,is.numeric), kurtosi)# kurtosis oko 3, za normal distribuciju

# graficki prikazi distribucija i zavisnosti nekih od prediktora sa zavisnom promenjivom

ggplot(default_cr, aes(x=MARRIAGE, y=LIMIT_BAL, color = default.payment.next.month))+geom_boxplot()
ggplot(default_cr, aes(x=SEX, y=LIMIT_BAL, color = default.payment.next.month))+geom_boxplot()
ggplot(default_cr, mapping = aes(x=AGE, y=LIMIT_BAL))+
  geom_boxplot(aes(fill = EDUCATION))+
  facet_wrap(~default.payment.next.month)


# 3. PROVERA POSTOJANJA MULTIKOLINEARNOSTI 


corelacija<-cor(select_if(default_cr,is.numeric))# korelacija prediktora
corrplot(corelacija,method='circle',type="upper",order="hclust",tl.col='black',tl.pos='n') # plava boja, pozitivna korelacija, crvena negativna. Intezitet boje odgovara intezitetu lin zavisnosti
corr_cross(select_if(default_cr,is.numeric), max_pvalue = 0.05, top = 20)# package lares. Parovi 10 znacajnih korelacija(p-value<0.05) U plavoj boji su pozitivne korelacije. Korelacije vece od 5%


# 4.KREIRANJE TRENING/TESTING PODATAKA 

# Train/test podaci u odnosu 70/30

set.seed(20)

split <- createDataPartition (default_cr$default.payment.next.month, p = .7, list = F)# caret
train <- norm_data[split,]
test  <- norm_data[-split,]

prop.table(table(train$default.payment.next.month))# nebalansirani podaci

# Balansiranje trening podataka

train_default<-train[train$default.payment.next.month =="Default",]
train_nodefault<-train[train$default.payment.next.month =="Not Default",]

train_nodefault<-train_nodefault[sample(nrow(train_nodefault),4642),]
train<-rbind(train_default,train_nodefault)

prop.table(table(train$default.payment.next.month))# balansirani trening podaci
prop.table(table(test$default.payment.next.month))# nebalansirani test podaci

train<-train[sample(1:nrow(train)),]

actuals<-test$default.payment.next.month# sacuvacemo originalne vrednosti za K NN i SVM, potrebne usled naknadnih transformacija u ovim algoritmima


# Odredjivanje znacaja, zavisnosti  prediktora van ML modela, primenom biserialne korelacije (Pearson, 1909) izmedju numerickih i kategorick

train_num<-select_if(train,is.numeric)
cor_pred_zavisna<-sapply(1:23, function(i) biserial.cor(train_num[,i],train$default.payment.next.month))
atributi<-colnames(train_num)
vaznost_attr<-data.frame(atributi,cor_pred_zavisna)
print(vaznost_attr[order(abs(vaznost_attr$cor_pred_zavisna), decreasing = TRUE), ][1:10,],row.names=F)


# 5. KREIRANJE PC


pc<-prcomp(train[,-24],center=T, scale=T) # kreiranje PC1...PC23

# pc$rotation  # linerna transformacija prediktora u PC. PC= f(x1,...,x23)
# pc$x # uzorci izrazeni u vrednostima PC, sada su prediktori PC1...PC23

get_eig(pc)[[1]][1:10]# prvih deset eigenvectors

#Kaiser's pravilo.Zadrzavaju se samo PC cije su eigenvectors veci od 1

predict_pctrain<-predict(pc, train[,-24])[, 1:5]# ne potrebno to je isto kao pc$x
train_pc<-data.frame(predict_pctrain, default.payment.next.month=train$default.payment.next.month)

predict_pctest<-predict(pc,test[,-24])[,1:5]
test_pc<-data.frame(predict_pctest, default.payment.next.month=test$default.payment.next.month)

cat("Broj PC sa vrednostima eigenvalue >= jedan je",max(which (get_eig(pc)[,1]>=1,)),"\n")
get_eig(pc)[[3]][1:5][5]# Kumulativna varijacija prvih 5 PC's (Cumulative proportion)

#  Korelacija izmedju PC je nula, nema linearn zavisnosti

corelacija_pc<-cor(pc$x)
corrplot(corelacija_pc,type="full",order="hclust",tl.col='black',tl.pos='n') # plava boja, pozitivna korelacija, crvena negativna. Intezitet boje odgovara intezitetu lin zavisnosti

#Train/test podaci u funkciji PCs


# 6. PREDIKTIVNI MODELI


# 6.1 LOGISTICKA REGRESIJA

# Kreiranje modela

model_lr<-glm(default.payment.next.month ~ ., data=train, family = binomial)
summary(model_lr)

anova(model_lr, test = 'Chisq')# test vaznosti prediktora

# Predikcija

predict_lr<-model_lr%>%predict(test,type="response")

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,predict_lr)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_lr",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)

predicted_lr<-as.factor(ifelse(predict_lr>0.530,"Default","Not Default"))
predicted_lr<-factor(predicted_lr, levels=rev(levels(predicted_lr)))

confusionMatrix(predicted_lr,test$default.payment.next.month, positive = 'Default')

# Model samo sa statisticki znacajnim prediktorima

model_lr1<-glm(default.payment.next.month ~ LIMIT_BAL+SEX+MARRIAGE+PAY_0+PAY_2+BILL_AMT1+PAY_AMT1+PAY_AMT2, data=train, family = binomial)
summary(model_lr1)

# Poredjenje dva modela primenom ANOVA test. Ho - drugi model je bolji od prvog.  Za p < 0.05, odbacujemo Ho

anova(model_lr,model_lr1,test = "Chisq")#

# Predikcija

predict_lr1<-model_lr1%>%predict(test,type="response")

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,predict_lr1)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_lr1",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_lr1<-as.factor(ifelse(predict_lr1>0.515,"Default","Not Default"))
predicted_lr1<-factor(predicted_lr1, levels=rev(levels(predicted_lr1)))
confusionMatrix(factor(predicted_lr1), test$default.payment.next.month, positive = "Default")


#6.1.1 Log reg sa PC

model_lrpc<-glm(default.payment.next.month ~ ., data=train_pc, family = binomial)#model sa PC
summary(model_lrpc)

# Predikcija

predict_lrpc<-model_lrpc%>%predict(test_pc,type="response")# sa PC

# Tacnost sa PC

res.pos<-roc(test_pc$default.payment.next.month,predict_lrpc)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_lrpc",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_lrpc<-as.factor(ifelse(predict_lrpc>0.568,"Default","Not Default"))

predicted_lrpc<-factor(predicted_lrpc, levels=rev(levels(predicted_lrpc)))
confusionMatrix(predicted_lrpc,test_pc$default.payment.next.month, positive = 'Default')


# 6.2 PENALIZED LOGISTIC REGRESSION, regularizacija modela (ne proredimo prema AIC)

#Lasso  regresija

# Kreiranje matrice prediktora 

x_train<- model.matrix(default.payment.next.month ~., data=train,)[,-24]# U slucaju postojanja factor promenjivih, kreiraju se dummy promenjive
x_test<-model.matrix(default.payment.next.month ~., test,)[,-24]

y_output<-train$default.payment.next.month

# Koriscenjem CV trazimo najoptimalniju vrednost lambda, kojom se odredjuje stepen 'sankcionisanja' kompleksnosti modela tj. umanjenje reg koeficijenata 

lasso_param<-cv.glmnet(x_train,y_output,alpha=1,family="binomial")# glmnet package.  Za alpha 0=1, radi se o Lasso reg

# Kreiranje modela

model_lrlasso<- glmnet(x_train,y_output, alpha = 1, family = "binomial",lambda = lasso_param$lambda.min)

# Predikcija

predict_lrlasso<-model_lrlasso%>%predict(newx=x_test,type="response")

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,as.numeric(predict_lrlasso))
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_lrlassoc",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_lrlasso<-as.factor(ifelse(predict_lrlasso>0.524,"Default","Not Default"))
predicted_lrlasso<-factor(predicted_lrlasso, levels=rev(levels(predicted_lrlasso)))
confusionMatrix(predicted_lrlasso, test$default.payment.next.month, positive = 'Default')


# Najvazniji, najinformativniji prediktori

coef(model_lrlasso)# 
df_regkoeficijenti<-data.frame(summary(coef(model_lrlasso)))[-1,]
df_regkoeficijenti$i<-df_regkoeficijenti$i-2
print(df_regkoeficijenti[order(-abs(df_regkoeficijenti$x)),][c(1:5),], row.names=FALSE) # 5 najvecih reg koeficijenata po apsolutnoj vrednosti 


# Ridge regresija

ridge_param<-cv.glmnet(x_train,y_output,alpha=0,family="binomial")# glmnet package.  Za alpha =0 , radi se o Ridge reg

# Kreiranje modela

model_lrridge<- glmnet(x_train,y_output, alpha = 0, family = "binomial",lambda = ridge_param$lambda.min)

# Predikcija

predict_lrridge<-model_lrridge%>%predict(newx=x_test,type="response")

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,as.numeric(predict_lrridge))
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_lrridge",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)



predicted_lrridge<-as.factor(ifelse(predict_lrridge>0.528,"Default","Not Default"))
predicted_lrridge<-factor(predicted_lrridge, levels=rev(levels(predicted_lrridge)))
confusionMatrix(predicted_lrridge, test$default.payment.next.month, positive = 'Default')

# Najvazniji, najinformativniji prediktori

coef(model_lrridge)# Intercept i 64 prediktora
df_regkoeficijenti<-data.frame(summary(coef(model_lrridge)))[-1,]# uklanjam intercept
df_regkoeficijenti$i<-df_regkoeficijenti$i-2
print(df_regkoeficijenti[order(-abs(df_regkoeficijenti$x)),][c(1:5),], row.names=FALSE) # 5 najvecih reg koeficijenata po apsolutnoj vrednosti 



# Elastic net regresija

elasticnet_param<-cv.glmnet(x_train,y_output,alpha=0.7,family="binomial")# glmnet package. Za alpha izmedju 0 i 1, radi se o Elastic Net regresiji

# Kreiranje modela

model_elasnet<- glmnet(x_train,y_output, alpha = 0.7, family = "binomial",lambda = elasticnet_param$lambda.min)

# Predikcija

predict_elasnet<-model_elasnet%>%predict(newx=x_test,type="response")

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,as.numeric(predict_elasnet))
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_elasnet",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)



predicted_elasnet<-as.factor(ifelse(predict_elasnet>0.54,"Default","Not Default"))
predicted_elasnet<-factor(predicted_elasnet, levels=rev(levels(predicted_elasnet)))

confusionMatrix(factor(predicted_elasnet), test$default.payment.next.month, positive = 'Default')

# Najvazniji, najinformativniji prediktori

coef(model_elasnet)# Intercept i 64 prediktora
df_regkoeficijenti<-data.frame(summary(coef(model_elasnet)))[-1,]# uklanjam intercept
df_regkoeficijenti$i<-df_regkoeficijenti$i-2
print(df_regkoeficijenti[order(-abs(df_regkoeficijenti$x)),][c(1:5),], row.names=FALSE) # 5 najvecih reg koeficijenata po apsolutnoj vrednosti 


# 6.3 LDA i QDA (koriscenjem prediktora najtacnijeg  lasso regresionog modela)


# LDA

model_lda<-lda(default.payment.next.month ~ ., data=train)
model_lda

par(mar=c(1,1,1,1))
plot(model_lda)# x osa je LD1, prvi grafik za klasu 0 , drugi klasu 1

# Predikcija

predict_lda<-model_lda%>%predict(test)
names(predict_lda)# class uzorka, posterior pr da uzorak pripda nekoj od klasa, x su LD
head(predict_lda$posterior,2)


# Tacnost modela

res.pos<-roc(test$default.payment.next.month,predict_lda$posterior[,2])
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_lad",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_lda<-as.factor(ifelse(predict_lda$posterior[,2] >0.52,"Default","Not Default"))
predicted_lda<-factor(predicted_lda, levels=rev(levels(predicted_lda)))

confusionMatrix(factor(predicted_lda), test$default.payment.next.month, positive = 'Default')


# Najvazniji, najinformativniji prediktori

coef(model_lda)# Intercept i 64 prediktora

df_regkoeficijenti<-data.frame(coef(model_lda))
impattr_lda<-rownames_to_column(df_regkoeficijenti)# package tibble
colnames(impattr_lda)<-c('Prediktori','RegKoeficijenti')
impattr_lda[order(-abs(impattr_lda$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# QDA

model_qda<-qda(default.payment.next.month ~ .,data=train)

# Predikcija

predict_qda<-model_qda%>%predict(test)

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,predict_qda$posterior[,2])
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_qda",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_qda<-as.factor(ifelse(predict_qda$posterior[,2] >0.982,"Default","Not Default"))
predicted_qda<-factor(predicted_qda, levels=rev(levels(predicted_qda)))
confusionMatrix(predicted_qda, test$default.payment.next.month, positive = 'Default')


# 6.4 STABLA ODLUCIVANJA

# Kreiranje modela

model_tree<-rpart(default.payment.next.month~., data=train, method = "class",control=rpart.control(cp=.0001)) # rpart koristi Ginin index kao meru imfomativnosti prediktora
# model_tree # zvezdica oznaka terminalnog noda, simbol ispred zagrade je sta taj nod predvidja ili sta bi bilo predvidjanje kad bi bio terminalni, broj posle uslova je broj uzorka

# Pruning stabla sa Optimalnom vrednosti cp,  odnosno optimalne dubine (kompleksnosnti) stabla.

plotcp(model_tree) 
best_cp <- model_tree$cptable[which.min(model_tree$cptable[,"xerror"]),"CP"]
cat("Optimalna vrednost cp je ",best_cp,"\n")

model_tree1<-prune(model_tree,cp=best_cp)# ako stavim 0.01 stablo ce biti isto kao u modelu model_tree.Biram manje kompleksno stablo
model_tree1

# Model stabla nakon pruninga
rpart.plot(model_tree1)

# najinfoirmativnijih prediktora sa najvecim  mean decrease accuracy 

impattr_tree1<-data.frame(model_tree1$variable.importance)
head(impattr_tree1,10)

# Predikcija primenom DT

predict_tree<-data.frame(predict(model_tree1,test,type="prob"))

# Tacnost modela

res.pos<-roc(test$default.payment.next.month,predict_tree[,2])
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_tree",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_tree <- as.factor(ifelse(predict_tree[,2] > 0.418, "Default","Not Default"))
predicted_tree<-factor(predicted_tree, levels=rev(levels(predicted_tree)))

confusionMatrix(predicted_tree,
                test$default.payment.next.month,positive = "Default")


# 6.5 RANDOM FOREST


trControl=trainControl("cv",number=5)# repeates, izostavljeno zbog performansi

# Kreiranje modela

model_rf<-train(default.payment.next.month~., data=train,method="rf",trControl= trControl,importance=TRUE, search='random')# Importance, vaznost atributa 
# plot(model_rf)

model_rf$bestTune 
cat("Optimalan broj prediktora koji ce se koristiti pri izradi stabala je" ,model_rf$finalModel$mtry )

model_rf$finalModel 

# 5 najinefirmativnijih prediktora sa najvecim  mean decrease accuracy 

imp_prediktora<-as.data.frame(randomForest::importance(model_rf$finalModel))

imp_prediktora[order(imp_prediktora$MeanDecreaseAccuracy, decreasing = T),][1:5,] 

# Graficki prikaz informativnosti prediktora

varImpPlot(model_rf$finalModel,type=1, cex=0.5, main = "Informativnost prediktora")# Mean decrease accuracy, koliko se smanji tacnost modela ako se dati prediktor izostavi

# Predikcija primenom RF modela

predict_rf<- predict(model_rf, test,type = "prob")

# Tacnost modela


res.pos<-roc(test$default.payment.next.month,predict_rf[,2])
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_rf",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_rf <- as.factor(ifelse(predict_rf[,2]> 0.495, "Default","Not Default"))
predicted_rf<-factor(predicted_rf, levels=rev(levels(predicted_rf)))

confusionMatrix(predicted_rf,
                test$default.payment.next.month,positive = "Default")


# 6.6 BOOSTING MODEL,  xgboost sa xgbTree. Koristi sve corove procesora, paralelno procesiranje


#define predictor and response variables in training set

train_x = data.matrix(train[, -24])
train_y =as.numeric(train[,24])-1


#define predictor and response variables in testing set
test_x = data.matrix(test[, -24])
test_y = as.numeric(test[,24])-1

#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

head(getinfo(xgb_train,'label'))
head(getinfo(xgb_test,'label'))


# Kreiranje modela

watchlist = list(train=xgb_train, test=xgb_test)#
train_par<- xgb.train(data = xgb_train, max.depth = 3, nrounds = 100, watchlist = watchlist)#training and testing root mean squared error for each round.

model_xgb<- xgboost(data = xgb_train, max.depth = 3, nrounds = 100, objective = "binary:logistic")# svi prediktori moraju biti numericki vektori

#  5 najinfoirmativnijih prediktora sa najvecim  mean decrease accuracy 

importance_matrica = xgb.importance(colnames(xgb_train), model = model_xgb)
importance_matrica[1:5,][,1]

xgb.plot.importance(importance_matrica[1:5,])

# Predikcija

predict_xgboost<-predict(model_xgb, newdata=xgb_test)

# Tacnost modela


res.pos<-roc(test$default.payment.next.month,predict_xgboost)

plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_xgb1",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_xgboost <- as.factor(ifelse(predict_xgboost> 0.546, "Default","Not Default"))
predicted_xgboost<-factor(predicted_xgboost, levels=rev(levels(predicted_xgboost)))


confusionMatrix(predicted_xgboost,
                test$default.payment.next.month,positive = "Default")


# 6.7 NB

# Kreiranje modela

model_nb<-naiveBayes(train$default.payment.next.month~., data=train)
model_nb # vidimo mean i sd za svaki numericki prediktor razlicitih klasa

train%>%filter(default.payment.next.month =="Default")%>%summarise(mean(PAY_AMT1),sd(PAY_AMT2))

# Predikcija

predict_nbp<-predict(model_nb,test,type='raw')[,2]

# Tacnost modela


res.pos<-roc(test$default.payment.next.month,predict_nbp)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_nb",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_nb <- as.factor(ifelse(predict_nbp> 0.909, "Default","Not Default"))
predicted_nb<-factor(predicted_nb, levels=rev(levels(predicted_nb)))


confusionMatrix(predicted_nb,
                test$default.payment.next.month,positive = "Default")


# 6.7.1 Model NB  sa PC

model_nbpc<-naiveBayes(default.payment.next.month~., data=train_pc)

# Predikcija

predict_nbpc<-predict(model_nbpc,test_pc,type='raw')[,2]# Tacnost modela

# Tacnost modela


res.pos<-roc(test_pc$default.payment.next.month,predict_nbpc)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_nbpc",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_nbpc <- as.factor(ifelse(predict_nbpc> 0.671, "Default","Not Default"))
predicted_nbpc<-factor(predicted_nbpc, levels=rev(levels(predicted_nbpc)))


confusionMatrix(predicted_nbpc,
                test_pc$default.payment.next.month,positive = "Default")


# 6.8 KNN model


# Primena cv, za trazenje optimalne vrednosti k

trControl<-trainControl(method = "repeatedcv", number = 5,repeats=3)

# Kreiranje modela 

model_knn<-train(default.payment.next.month ~., data = train, method='knn', 
                 trControl=trControl,
                 tuneLength=10, preProcess=c("center","scale"))# teuneLength broj mogucih vrednosti  K 


model_knn$bestTune# optimalna vrednost suseda k

plot(model_knn,xlab="Broj suseda K",ylab='Tacnost u funkciji od K', type='b',col='red',lwd=1.5,pch='o')

# Najznacajniji prediktori za tacnost modela

impattr_knn<-data.frame(varImp(model_knn)[[1]])
impattr_knn<-rownames_to_column(impattr_knn)# package tibble

colnames(impattr_knn)<-c('Prediktori','RegKoeficijenti')
impattr_knn[order(-abs(impattr_knn$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

#Predikcija

predict_knn<-predict(model_knn, test, type = "prob")[,2] # X1 stecaj


# Tacnost modela

res.pos<-roc(test$default.payment.next.month,predict_knn)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_knn",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_knn = as.factor(ifelse(predict_knn > .538, "Default","Not Default"))
predicted_knn<-factor(predicted_knn, levels=rev(levels(predicted_knn)))

confusionMatrix(predicted_knn,actuals, positive = "Default")



# 6.9  SUPORT VECTOR MACHINE linear


#kreiranje sintaticka ispravnih imena, za vrednosti karakter vektora


levels(train$default.payment.next.month)<-make.names(levels(train$default.payment.next.month))
levels(test$default.payment.next.month)<-make.names(levels(test$default.payment.next.month))



trControl<-trainControl(method = "cv", number = 5, summaryFunction=twoClassSummary,classProbs=TRUE)#za izbor modela na bazi ROC

# Kreiranje  modela, SVM sa linearnom Kernel funkcijom

model_svmlin <- train(default.payment.next.month ~ ., data = train, method = "svmLinear", preProcess=c("center","scale"), trControl=trControl)# tuneLength = 5,tuneGrid = grid)#,metric="ROC"

# Prikaz modela

model_svmlin # c=1, za lin metod.Ova konstanta predstavlja cost, misklasifikacije. Tako se sa vecom
# vrednosti konstante C, verovatnoca pogresne klasifikacije je manja.


# Najznacajniji prediktori za tacnost modela

impattr_svmlin<-data.frame(varImp(model_svmlin)[[1]])
impattr_svmlin<-rownames_to_column(impattr_svmlin)
impattr_svmlin[order(-abs(impattr_svmlin[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmlin<-predict(model_svmlin,test, type='prob')[,2]

res.pos<-roc(test$default.payment.next.month, predict_svmlin)

plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_svmlin",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)


predicted_svmlin <- factor(ifelse( predict_svmlin > 0.567, "Default","Not Default"))
predicted_svmlin<-factor(predicted_svmlin, levels=rev(levels(predicted_svmlin)))
confusionMatrix(predicted_svmlin,actuals,positive = "Default" )

# SVM Radial.  Kreiranje modela, SVM sa nelinearnom kernel funkcijom (Radial)

model_svmradial <- train(default.payment.next.month ~., data = train, method = "svmRadial", preProcess=c("center","scale") ,trControl=trControl)#, tuneLength = 5,metric="ROC")

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmradial$bestTune

# Prikaz modela

plot(model_svmradial)

# Najznacajniji prediktori za tacnost modela

impattr_svmradial<-data.frame(varImp(model_svmradial)[[1]])
impattr_svmradial<-rownames_to_column(impattr_svmradial)
impattr_svmradial[order(-abs(impattr_svmradial[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmradial<-predict(model_svmradial, test, type = "prob")[,2]

res.pos<-roc(test$default.payment.next.month, predict_svmradial)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_svmradial",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)

predicted_svmradial <-as.factor(ifelse( predict_svmradial> 0.554, "Default","Not Default"))
predicted_svmradial<-factor(predicted_svmradial, levels=rev(levels(predicted_svmradial)))
confusionMatrix(predicted_svmradial, actuals, positive = "Default")

#  SVM Poli

# Kreiranje modela (sa ogranicenim brojem uzoraka zbog performansi), SVM sa nelinearnom kernel funkcijom (polinomalna)

model_svmpoli <- train(default.payment.next.month ~ ., data = train[1:4000,], method = "svmPoly",preProcess=c("center","scale"),trControl=trControl)

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmpoli$bestTune

# Prikaz modela

plot(model_svmpoli)

# Najznacajniji prediktori za tacnost modela

impattr_svmpoli<-data.frame(varImp(model_svmpoli)[[1]])
impattr_svmpoli<-rownames_to_column(impattr_svmpoli)
impattr_svmpoli[order(-abs(impattr_svmpoli[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmpoli<-predict(model_svmpoli,test, type = "prob")[,2]

res.pos<-roc(test$default.payment.next.month, predict_svmpoli)
plot.roc(res.pos,print.auc = T, print.thres = "best", lwd = 1,asp = NA,auc.polygon=T,lty=3,cex.lab=.5,cex.axis=0.5,cex.main=0.8, main="ROC model_svmpoli",
         col="red",identity.col="red",print.thres.col="red",print.auc.adj=c(0,5),print.thres.adj=-.10)

predicted_svmpoli <-as.factor(ifelse( predict_svmpoli> 0.527, "Default","Not Default"))
predicted_svmpoli<-factor(predicted_svmpoli, levels=rev(levels(predicted_svmpoli)))

confusionMatrix(predicted_svmpoli,actuals, positive = "Default")



# Vracanje warn messages

options(warn = tek_seting)

# Kraj


