document.addEventListener('DOMContentLoaded', function() {
  // Your code here

  // Example:
  const form = document.querySelector('form');
  const firstName = document.getElementById("first_name");
  const lastName = document.getElementById("last_name");
  const email = document.getElementById("email");
  const phone = document.getElementById("phone");
  const message = document.getElementById("message");

  function sendEmail() {
      const bodymessage = `First Name: ${firstName.value} Last Name: ${lastName.value} <br> Email: ${email.value} <br> Phone: ${phone.value} <br> Message: ${message.value}`;
      Email.send({
        Host : "smtp.elasticemail.com",
        Username : "abhilashvisakan24@gmail.com",
        Password : "965467F816561EACF92D5EFCE9A3D0F326E5",
        To : 'abhilashvisakan24@gmail.com',
        From : "abhilashvisakan24@gmail.com",
        Subject : "This is the subject",
        Body : bodymessage,
        port:25,
      }).then(
          message => alert(message)
      );
  }

  form.addEventListener("submit", (e) => {
      e.preventDefault();
      sendEmail();
  });
});
