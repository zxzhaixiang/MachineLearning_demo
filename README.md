# MachineLearning_demo
repos for some machine learning demo notebooks. All the notebooks should be able to run directly on Google Colab. You might need to have a Google Drive account in order to run some of the notebooks.

## A trick to prevent Google Colab Timeout
If you leave the Colab notebook to run for too long without attention, Colab might decide to timeout and you will have to reconnect the the runtime. You might lose your progress. To avoid this and be able to use Colab for a straight 12hr (granted by Google), you can do the following trick in the browser:

- Press F12 in the browser to open JavaScript console
- Paste the following code to console
```
function ClickConnect(){console.log("Working");document.querySelector("colab-toolbar-button#connect").click()}
setInterval(ClickConnect,60000)
```
This will keep Colab active until your session is ended.
