var socket = io.connect('http://localhost:5000');

socket.on('detection', function (data) {
    var numFish = data.num_fish;
    document.getElementById('num-fish').innerText = numFish;

    // Uyarı göster (3'ten fazla aslan balığı tespit edilirse)
    if (numFish > 3) {
        document.getElementById('alert').innerText = "Uyarı: Çok fazla aslan balığı tespit edildi!";
    } else {
        document.getElementById('alert').innerText = "";
    }
});