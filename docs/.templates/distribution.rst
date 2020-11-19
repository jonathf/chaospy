{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

      ~Distribution.pdf
      ~Distribution.cdf
      ~Distribution.fwd
      ~Distribution.inv
      ~Distribution.sample
      ~Distribution.mom
      ~Distribution.ttr

   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::

      ~Distribution.interpret_as_integer
      ~Distribution.lower
      ~Distribution.stochastic_dependent
      ~Distribution.upper

   {% endif %}
   {% endblock %}
