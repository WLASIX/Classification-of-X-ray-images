I will highlight a few observations about this project:
1. training is CPU-only, so training via Grid Search will take a long time. But essentially scikit-learn due to this allows you to run the training on any pc.
2. The dataset was too varied yet small. It was not possible to find pictures of fractures of certain body parts in sufficient quantity, so the accuracy of the model suffers a lot.
3. The model performs poorly on the test sample. To increase the accuracy of the model, it was decided to augment the original data (reflect vertically, flip 45 degrees).
4. There is practically no difference in metrics between Random Forest and Grid Search, respectively the parameters of Random Forest are selected accurately. However, everything will depend on which images were taken by the program as training images (for RF)

-------------------------------------------------------

Выделю несколько замечаний по этому проекту:
1. Обучение происходит только на процессоре, поэтому обучение с помощью Grid Search займет много времени. Но, по сути, scikit-learn благодаря этому позволяет запускать обучение на любом ПК.
2. Набор данных был слишком разнообразным и в то же время небольшим. Не удалось найти снимки переломов определенных частей тела в достаточном количестве, поэтому точность модели сильно страдает.
3. Модель плохо работает на тестовой выборке. Чтобы повысить точность модели, было решено дополнить исходные данные (отразить по вертикали, перевернуть на 45 градусов).
4. Разницы в метриках между Random Forest и Grid Search практически нет, соответственно параметры Random Forest подобраны точно. Однако все будет зависеть от того, какие изображения были взяты программой в качестве обучающих (для RF)
